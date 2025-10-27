"""The 5 sandbox cases of the plan's checklist (section 17): normal, timeout, exception, no output, and a second
call within one conversation (through the engine's tool loop with a scripted model); plus pagoda's executor
checks (isolation, memory cap, truncation, answer parsing, bulk execution) and the Qwen3 tool-call wrapper."""

import pytest

from bonsai import chat as C
import time

from bonsai import sandbox as S
from bonsai.sandbox import (extract_answer, fast_exec_many, format_number, is_safe_code, numbers_equal,
                            repl_source, run_python, run_tool_call, single_number)


def call(code):
    return C.python_call(code)


def test_normal():
    r = run_python("total = 3 * 12\nprint(total - 10)\n")
    assert r.ok and r.output == "26"
    assert run_tool_call(call("total = 3 * 12\nprint(total - 10)")) == "26"


def test_timeout():
    r = run_python("while True:\n    pass\n")
    assert not r.ok and r.timed_out and r.output.startswith("TimeoutError")
    assert run_tool_call(call("import time\ntime.sleep(10)")).startswith("TimeoutError")


def test_exception():
    r = run_python("x = 1 / 0\n")
    assert not r.ok and r.output == "ZeroDivisionError: division by zero"
    assert run_tool_call(call("print(100 / 0)")) == "ZeroDivisionError: division by zero"
    out = run_tool_call(call("print('partial')\nraise ValueError('bad value')"))
    assert out == "partial\nValueError: bad value"   # stdout so far + the error line, verbatim


def test_no_output():
    r = run_python("x = 5\n")
    assert r.ok and r.output == ""
    assert run_tool_call(call("x = 5")) == ""


def test_second_call_in_one_conversation():
    """Two tool calls in one conversation via engine.chat: each runs in a fresh process, both results come back
    in order, the final answer uses the second one."""
    from bonsai import engine as E
    from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap
    qwen = QwenTokenizer()
    tok = Tokenizer(qwen, VocabMap.for_tests(qwen))
    conv = [{"role": "user", "content": "Compute 3 * 12 - 10, then add 4. /no_think"},
            {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [call("print(3 * 12 - 10)")]},
            {"role": "tool", "content": "26"},
            {"role": "assistant", "content": "", "reasoning_content": "",
             "tool_calls": [call("x = 26\nprint(x + 4)")]},
            {"role": "tool", "content": "30"},
            {"role": "assistant", "content": "Answer: 30", "reasoning_content": ""}]
    target = [tok.bos_id] + tok.from_teacher(C.render(qwen, conv, tools=C.TOOLS)[0]).tolist()
    end = tok.special("<|im_end|>")
    stub = E.ScriptedModel(E.scripted_policy([target], end), tok.vocab_size, tok.bos_id)
    r = E.chat(stub, tok, conv[:1], tools=C.TOOLS)
    assert r.n_tool_calls == 2 and [c["output"] for c in r.tool_calls] == ["26", "30"]
    assert r.finished and extract_answer(r.text) == 30 and r.ids == target[:-1]


def test_multiple_prints():
    r = run_python("print(1)\nprint(2)\n")
    assert r.ok and r.output == "1\n2"
    assert single_number(r.output) is None


def test_output_truncated():
    r = run_python("print('a' * 1000)")
    assert len(r.output) == 256 and r.output.endswith("...")


def test_no_network_no_writes_outside():
    r = run_python("import socket\ns = socket.create_connection(('1.1.1.1', 80), timeout=1)\nprint('connected')")
    assert not r.ok and "connected" not in r.output
    r = run_python("open('/tmp/bonsai_sandbox_escape.txt', 'w').write('x')\nprint('wrote')")
    assert not r.ok and "wrote" not in r.output


def test_large_output_is_drained():
    """More than the 64 KB pipe buffer: read while the child runs, so it finishes (not a false timeout)."""
    t0 = time.monotonic()
    r = run_python("print('x' * 70000)")
    assert r.ok and not r.timed_out and len(r.output) == 256 and time.monotonic() - t0 < 1.5
    r = run_python("for i in range(20000):\n    print(i)")
    assert r.ok and r.output.startswith("0\n1\n2") and len(r.stdout) == S.CAPTURE


def test_no_fork_and_no_wait_for_escaped_children(monkeypatch):
    assert run_python("import os\nos.fork()\nprint('forked')").output.startswith("BlockingIOError")
    # without the fork limit, a child in its own session keeps the pipes open: not waited for
    monkeypatch.setattr(S, "_ULIMITS", S._ULIMITS.replace("ulimit -u 1; ", ""))
    code = "import os, time\nif os.fork() == 0:\n    os.setsid()\n    time.sleep(5)\n    os._exit(0)\ntime.sleep(4)"
    t0 = time.monotonic()
    r = run_python(code, timeout=1.0)
    assert r.timed_out and time.monotonic() - t0 < 2.5


def test_repl_semantics():
    """The tool shows the repr of a final bare expression after the program's own output, like a notebook cell;
    a program ending in print(...) is not touched; run_python alone keeps plain script semantics."""
    cases = {"3500 / 500": "7.0", "x = 5; x + 1  # six": "6", "print('a')\n10": "a\n10", "'hi'": "'hi'",
             "x = 3\n(x *\n  2)": "6", "def f():\n    return 3\nf()": "3", "a = [1]\na.append(2)": "",
             "y = None\ny": "", "1 / 0": "ZeroDivisionError: division by zero", "x = 4\nx  # é": "4"}
    for code, want in cases.items():
        assert run_tool_call(call(code)) == want, code
    for code in ("total = 3 * 12\nprint(total - 10)", "print(1)\nprint(2)\n", "x = 5\n", "while True:\n    pass\n",
                 "if True:\n    7", "print(("):
        assert repl_source(code) == code
    assert run_python("3500 / 500").output == "" and run_python("3500 / 500", repl=True).output == "7.0"


def test_tool_call_never_raises(monkeypatch):
    """A lone surrogate (half an emoji pair in model JSON) keeps its meaning instead of crashing the write; an
    executor failure (disk full, EAGAIN) comes back as an error string."""
    calls = C.parse_tool_calls('<tool_call>\n{"name": "python", "arguments": {"code": "print(\\"\\ud83d\\")"}}'
                               '\n</tool_call>')
    assert calls[0]["arguments"]["code"] == 'print("\ud83d")'
    assert run_tool_call(calls[0]).startswith("UnicodeEncodeError")
    assert run_tool_call(call('s = "\ud83d"\nprint(len(s))')) == "1"

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(S, "run_python", boom)
    out = run_tool_call(call("print(1)"))
    assert out.startswith("Error: OSError") and "No space left" in out and len(out) <= S.MAX_OUTPUT


def test_memory_cap():
    r = run_python("x = bytearray(900 * 1024 * 1024)\nfor i in range(0, len(x), 4096): x[i] = 1\nprint('done')",
                   timeout=5)
    assert not r.ok and "done" not in r.output


@pytest.mark.parametrize("bad,msg", [({"name": "search", "arguments": {"q": "x"}}, "unknown tool"),
                                     ({"name": "python", "arguments": {}}, "non-empty 'code'"),
                                     ({"name": "python", "arguments": "print(1)"}, "non-empty 'code'"),
                                     ("not a dict", "unknown tool")])
def test_malformed_calls(bad, msg):
    assert msg in run_tool_call(bad)


def test_parse_tool_calls_roundtrip():
    text = '<tool_call>\n{"name": "python", "arguments": {"code": "print(2 + 3)"}}\n</tool_call>'
    calls = C.parse_tool_calls(text)
    assert calls == [call("print(2 + 3)")] and run_tool_call(calls[0]) == "5"


def test_answer_parsing():
    assert extract_answer("blah\nAnswer: 26") == 26
    assert numbers_equal(extract_answer("Answer: 26.0"), 26)
    assert extract_answer("Answer: $1,234.50") == 1234.5
    assert extract_answer("Answer: 3\nAnswer: -4") == -4
    assert extract_answer("no answer here") is None
    assert format_number(26.0) == "26" and format_number(2.5) == "2.5"
    assert single_number("26.0") == 26.0 and single_number("26 apples") is None


def test_fast_exec_many():
    codes = ["print(2 + 3)", "import os\nprint(os.getcwd())", "x = 1/0", "print(10 ** 100000)"]
    assert not is_safe_code(codes[1]) and not is_safe_code(codes[3])
    out = fast_exec_many(codes, workers=2)
    assert out[0] == "5\n" and out[1] is None and out[2] is None and out[3] is None
