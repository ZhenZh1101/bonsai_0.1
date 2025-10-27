"""Python executor for the math tool (plan section 13; ported from pagoda).

run_python(code):
  * `python -I -S -B main.py` in a fresh empty dir under sandbox/, stdin closed; main.py is written as UTF-8 with
    lone surrogates (half an emoji pair in model JSON) backslash-escaped, so a string literal keeps its meaning
  * macOS `sandbox-exec` profile: no network, no file writes outside that dir
  * 2 s wall-clock timeout (whole process group killed), CPU rlimit 3 s, 1 MB files, no core dumps, no fork (the
    user's process limit is set to 1); the limits are set by `/bin/sh -c ulimit ...; exec python` in a new session
    (no preexec_fn: Popen with a preexec_fn is not safe while other threads run, and the engine runs tool calls
    from a thread pool)
  * 512 MB memory cap: macOS does not enforce RLIMIT_AS / `ulimit -v`, so a
    watchdog polls the child's RSS and kills it above the cap
  * stdout / stderr are read while the child runs (first 64 KB kept), so a large output neither blocks the child
    nor reads as a timeout; after the child exits, whatever still holds the pipes is not waited for
  * output truncated to 256 chars; on an exception the error line is returned
    verbatim so the model can see it and retry
  * repl=True (the tool): notebook semantics, the repr of a final bare expression is printed after the program's
    own output unless it is None (`3500 / 500` shows 7.0; a program ending in print(...) runs unchanged)

Also: answer parsing / numeric comparison, `fast_exec_many` for bulk filtering
of millions of synthetic snippets (TinyGSM) in a process pool after an AST
allowlist check, and `run_tool_call` for a parsed Qwen3 `<tool_call>` (never raises: an error is a string for
the model; the same function for teacher generation, the student engine and the evals).
"""

import ast
import io
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass

from bonsai.common import SANDBOX_DIR

TIMEOUT_S = 2.0
MEM_LIMIT = 512 * 1024 * 1024
MAX_OUTPUT = 256
CAPTURE = 1 << 16    # bytes of stdout / stderr kept (the rest is read and dropped)
PYTHON = sys.executable

_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr")
                   (literal "/dev/tty") (literal "/dev/dtracehelper"))
"""


@dataclass
class ExecResult:
    output: str          # what is fed back inside <tool_response> ... </tool_response>
    ok: bool             # exited 0 without timeout / memory kill
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""


# CPU soft 3 s / hard 4 s, file size 1 MB (2048 blocks of 512 bytes in POSIX mode), no core files, no new processes
# (RLIMIT_NPROC counts all of the user's processes: 1 makes every fork fail, so nothing outlives the process group)
_ULIMITS = 'ulimit -S -t 3; ulimit -H -t 4; ulimit -f 2048; ulimit -c 0; ulimit -u 1; exec "$0" "$@"'


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _error_line(stderr):
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else "Error"


def truncate(s, n=MAX_OUTPUT):
    return s if len(s) <= n else s[: n - 3] + "..."


def _is_print(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"


def repl_source(code):
    """`code` with notebook semantics: a final top-level bare expression (not a print(...) call) is replaced by
    `__repl_value__ = (expr)` + printing its repr unless it is None. Everything before it is kept verbatim; code
    that does not parse, or needs no change, comes back unchanged."""
    try:
        tree = ast.parse(code)
        last = tree.body[-1] if tree.body else None
        if not isinstance(last, ast.Expr) or _is_print(last.value):
            return code
        starts = [0] + [m.end() for m in re.finditer(r"\r\n|\r|\n", code)]
        a = starts[last.lineno - 1]
        head = code[a:].encode("utf-8")[:last.col_offset].decode("utf-8")   # col_offset counts UTF-8 bytes
        expr = ast.get_source_segment(code, last)
        if expr is None:
            return code
        return (code[:a] + head + f"__repl_value__ = ({expr})\n"
                "if __repl_value__ is not None:\n    print(repr(__repl_value__))\n")
    except Exception:   # run the code as written: the child reports what is wrong
        return code


def _drain(pipes, bufs, timeout):
    """Read what the open pipes have within `timeout` (keeping the first CAPTURE bytes of each); EOF closes one."""
    if not pipes:
        time.sleep(timeout)
        return
    ready, _, _ = select.select(list(pipes), [], [], timeout)
    for fd in ready:
        chunk = os.read(fd, 1 << 16)
        if not chunk:
            del pipes[fd]
            continue
        b = bufs[pipes[fd]]
        if len(b) < CAPTURE:
            b += chunk[:CAPTURE - len(b)]


def run_python(code, timeout=TIMEOUT_S, mem_limit=MEM_LIMIT, use_sandbox_exec=True, repl=False):
    import psutil

    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="run_", dir=SANDBOX_DIR))
    try:
        src = code.encode("utf-8", "backslashreplace").decode("utf-8")   # a lone surrogate -> a \udxxx escape
        with open(os.path.join(workdir, "main.py"), "w", encoding="utf-8") as f:
            f.write(repl_source(src) if repl else src)
        cmd = ["/bin/sh", "-c", _ULIMITS, PYTHON, "-I", "-S", "-B", "main.py"]
        if use_sandbox_exec and os.path.exists("/usr/bin/sandbox-exec"):
            cmd = ["/usr/bin/sandbox-exec", "-p", _PROFILE.format(workdir=workdir)] + cmd
        env = {"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8", "HOME": workdir, "TMPDIR": workdir}
        proc = subprocess.Popen(cmd, cwd=workdir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, start_new_session=True)
        pipes, bufs = {proc.stdout.fileno(): 0, proc.stderr.fileno(): 1}, [bytearray(), bytearray()]
        deadline = time.monotonic() + timeout
        timed_out = mem_killed = False
        ps = None
        try:
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    _kill(proc)
                    break
                try:
                    ps = ps or psutil.Process(proc.pid)
                    rss = sum(p.memory_info().rss for p in [ps] + ps.children(recursive=True))
                    if rss > mem_limit:
                        mem_killed = True
                        _kill(proc)
                        break
                except psutil.Error:
                    pass
                _drain(pipes, bufs, 0.005)
            end = time.monotonic() + 0.5   # the rest of the output; a pipe still held open is given up on
            while pipes and time.monotonic() < end:
                _drain(pipes, bufs, max(end - time.monotonic(), 0.0))
            proc.wait()
        finally:
            proc.stdout.close()
            proc.stderr.close()
        out = bytes(bufs[0]).decode("utf-8", "replace")
        err = bytes(bufs[1]).decode("utf-8", "replace")
        if timed_out:
            return ExecResult(f"TimeoutError: execution exceeded {timeout:g} seconds", False, True, out, err)
        if mem_killed:
            return ExecResult("MemoryError: exceeded 512 MB", False, False, out, err)
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL) and not err.strip():
            return ExecResult("TimeoutError: CPU time limit exceeded", False, True, out, err)
        if proc.returncode != 0:
            msg = _error_line(err)
            text = (out.strip() + "\n" + msg) if out.strip() else msg
            return ExecResult(truncate(text), False, False, out, err)
        return ExecResult(truncate(out.strip()), True, False, out, err)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------- answers
_ANSWER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)")
_NUMBER_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?([eE][-+]?\d+)?$|^-?\.\d+$")


def parse_number(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").rstrip(".")
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text):
    """Number after the last 'Answer:' in a completion."""
    m = _ANSWER_RE.findall(text)
    return parse_number(m[-1]) if m else None


def numbers_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def single_number(output):
    """The printed output if it is exactly one number, else None."""
    s = output.strip()
    if not s or "\n" in s:
        return None
    s2 = s.replace(",", "")
    return parse_number(s2) if _NUMBER_RE.match(s2) else None


def format_number(x):
    if x is None:
        return ""
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------ bulk execution
SAFE_IMPORTS = {"math", "fractions", "decimal", "datetime", "itertools", "functools", "statistics",
                "collections", "string", "operator", "re"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "exit", "quit",
                "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "help"}


def is_safe_code(code):
    """Static allowlist used before in-process execution and as a data-quality filter."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] not in SAFE_IMPORTS for a in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_IMPORTS or node.level:
                return False
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return False
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.LShift)):
            # huge integer powers run in C and cannot be interrupted by the alarm
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                    and abs(node.right.value) <= 64):
                return False
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.Lambda)):
            return False
    return True


def _alarm(signum, frame):
    raise TimeoutError("timeout")


def _pool_init():
    os.chdir(tempfile.mkdtemp(prefix="bulk_", dir=SANDBOX_DIR))
    signal.signal(signal.SIGALRM, _alarm)


def _exec_one(code):
    if not is_safe_code(code):
        return None
    buf = io.StringIO()
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<snippet>", "exec"), {"__name__": "__main__"})
    except BaseException:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    out = buf.getvalue()
    return out if len(out) <= 4096 else None


def _run_isolated(code):
    if not is_safe_code(code):
        return None
    r = run_python(code, timeout=1.0)
    return r.stdout if r.ok and len(r.stdout) <= 4096 else None


def fast_exec_many(codes, workers=None, chunksize=256, batch=20000, batch_timeout=600):
    """stdout of each snippet (None on error / unsafe / timeout). Pure-arithmetic code only.

    Snippets run in-process inside a spawn pool. If a batch hangs (an operation the
    alarm cannot interrupt), the pool is killed and that batch is redone one snippet
    at a time in the subprocess sandbox.
    """
    import multiprocessing as mp
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    n_workers = workers or max(1, os.cpu_count() - 2)
    results = []
    pool = ctx.Pool(n_workers, initializer=_pool_init, maxtasksperchild=20000)
    try:
        for i in range(0, len(codes), batch):
            part = codes[i: i + batch]
            try:
                results.extend(pool.map_async(_exec_one, part, chunksize=chunksize).get(timeout=batch_timeout))
            except mp.TimeoutError:
                pool.terminate()
                pool = ctx.Pool(n_workers, initializer=_pool_init, maxtasksperchild=20000)
                results.extend(_run_isolated(c) for c in part)
    finally:
        pool.terminate()
        for d in SANDBOX_DIR.glob("bulk_*"):  # worker scratch dirs
            shutil.rmtree(d, ignore_errors=True)
    return results


# ------------------------------------------------------------ Qwen3 tool protocol


def run_tool_call(call, timeout=TIMEOUT_S):
    """Output string for one parsed tool call {"name": "python", "arguments": {"code": ...}} (plan section 13), run
    with notebook semantics (run_python(repl=True)). Never raises: unknown tools, malformed arguments and executor
    failures (an OSError such as a full disk) come back as an error string, like an exception would, so the model
    sees what went wrong and a sampling loop keeps going."""
    if not isinstance(call, dict) or call.get("name") != "python":
        name = call.get("name") if isinstance(call, dict) else None
        return f"Error: unknown tool {name!r}; the only tool is 'python'"
    args = call.get("arguments")
    code = args.get("code") if isinstance(args, dict) else None
    if not isinstance(code, str) or not code.strip():
        return "Error: the python tool needs a non-empty 'code' string"
    try:
        res = run_python(code, timeout=timeout, repl=True)
    except Exception as e:
        return truncate(f"Error: {type(e).__name__}: {e}").encode("utf-8", "replace").decode("utf-8")
    return res.output if res.output else ("" if res.ok else "Error")
