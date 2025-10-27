"""Qwen3 tokenizer, unchanged, plus the student's vocabulary subset (plan section 6).

Two id spaces:

  teacher ids  Qwen3's own ids, 0..151935 (151,643 BPE tokens + 26 control tokens, rows padded to 151,936)
  student ids  0..N-1 (N <= 65,536), the rows of the student's embedding; `s2t[student] = teacher`,
               `t2s[teacher] = student` or -1 when the teacher token is outside the subset S

Every split is done by Qwen3's tokenizer (tokenizer/tokenizer.json, pinned commit, sha256 in
tokenizer/manifest.json), so teacher and student see the same token sequence. The subset only changes the
lookup table: a teacher token outside S is written as its UTF-8 bytes, each a byte token that is always in S
(lossless; these positions are flagged so distillation can mask them).

Ordinary text is encoded with the control tokens removed from the tokenizer, so a literal "<|im_start|>" or
"<think>" in a document or a user turn becomes ordinary pieces ("<|", "im", "_start", "|>") and can never forge a
control token. For text without such literals this is exactly Qwen's own `encode` (acceptance check 2).

`Tokenizer` (student ids) is what training, data building and generation use; `QwenTokenizer` (teacher ids) is
what the chat renderer, the vocab-subset measurement and the teacher side of distillation use.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from bonsai.common import TOKENIZER_DIR

QWEN_REPO = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"   # pinned commit (plan section 6)
TEACHER_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
TEACHER_VOCAB = 151936      # rows of Qwen3's embedding / lm_head
N_BPE = 151643              # ordinary (BPE) tokens, ids 0..151642

# the 26 control tokens, ids 151643..151668 in this order; all are kept in the subset with their teacher ids
CONTROL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>",
    "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>",
    "</tool_response>", "<think>", "</think>",
]
CONTROL_IDS = {name: N_BPE + i for i, name in enumerate(CONTROL_TOKENS)}
BOS = "<|endoftext|>"   # Qwen3 has no BOS token; <|endoftext|> starts every student document / conversation
VOCAB_FILE = "vocab_subset.npz"


def _bytes_to_unicode():
    """GPT-2 / Qwen byte-level alphabet: byte value -> the unicode char used in vocab.json."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class QwenTokenizer:
    """Qwen3's tokenizer in teacher ids. `encode` never produces control tokens; `encode_qwen` is Qwen's own."""

    def __init__(self, directory=TOKENIZER_DIR):
        from tokenizers import Tokenizer as HFTokenizer

        self.dir = Path(directory)
        path = self.dir / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m scripts.download --only tokenizer` first")
        spec = json.loads(path.read_text())
        self.full = HFTokenizer.from_str(json.dumps(spec))
        plain = dict(spec)
        plain["added_tokens"] = []
        self.plain = HFTokenizer.from_str(json.dumps(plain))
        for name, tid in CONTROL_IDS.items():
            got = self.full.token_to_id(name)
            if got != tid:
                raise ValueError(f"control token {name}: tokenizer.json says {got}, expected {tid}")
        vocab = spec["model"]["vocab"]
        if len(vocab) != N_BPE:
            raise ValueError(f"expected {N_BPE} BPE tokens, found {len(vocab)}")
        b2u = _bytes_to_unicode()
        u2b = {u: b for b, u in b2u.items()}
        self.byte_ids = np.array([vocab[b2u[b]] for b in range(256)], dtype=np.int32)  # byte value -> teacher id
        # UTF-8 bytes of every teacher id (control tokens: their literal text, flagged separately)
        self._tok_bytes = [b""] * TEACHER_VOCAB
        for piece, tid in vocab.items():
            self._tok_bytes[tid] = bytes(u2b[c] for c in piece)
        for name, tid in CONTROL_IDS.items():
            self._tok_bytes[tid] = name.encode()
        self.control_ids = CONTROL_IDS

    # --- encoding -------------------------------------------------------------
    def encode(self, text):
        """Teacher ids of ordinary text; control-token literals are split into ordinary pieces."""
        return self.plain.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        fast = getattr(self.plain, "encode_batch_fast", None) or self.plain.encode_batch
        return [e.ids for e in fast(list(texts), add_special_tokens=False)]

    def encode_offsets(self, text):
        """(ids, offsets) for ordinary text; offsets are (start, end) character spans of `text`."""
        e = self.plain.encode(text, add_special_tokens=False)
        return e.ids, e.offsets

    def encode_qwen(self, text):
        """Exactly Qwen's own tokenization (control-token literals become control tokens)."""
        return self.full.encode(text, add_special_tokens=False).ids

    def encode_qwen_batch_offsets(self, texts):
        return [(e.ids, e.offsets) for e in self.full.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids, skip_special=False):
        ids = [int(i) for i in ids]
        if skip_special:
            ids = [i for i in ids if i < N_BPE]
        return self.full.decode(ids, skip_special_tokens=False)

    def token_bytes(self, tid):
        return self._tok_bytes[int(tid)]

    def byte_lengths(self):
        """UTF-8 byte length of every teacher id; 0 for control tokens and padding rows (bits per byte)."""
        out = np.array([len(b) for b in self._tok_bytes], dtype=np.int32)
        out[N_BPE:] = 0
        return out

    def has_control_literal(self, text):
        return any(name in text for name in CONTROL_TOKENS) if "<" in text else False


class VocabMap:
    """The student's subset S of teacher ids: s2t (len N), t2s (len 151,936, -1 outside S).

    A teacher token outside S is rewritten losslessly as in-S tokens with the same bytes: `fallback="pieces"`
    (default) uses the fewest in-S pieces (" Mississippi" -> " Miss" "iss" "ippi"), `"bytes"` the single bytes
    (the plan's literal rule; 4-8x longer). Either way the decoded text is identical and the positions are flagged.
    Shards are uint16 when N <= 65,536, else uint32 (`shard_dtype`)."""

    def __init__(self, s2t, qwen: QwenTokenizer, meta=None, fallback=None):
        self.qwen = qwen
        self.s2t = np.asarray(s2t, dtype=np.int32)
        self.N = len(self.s2t)
        self.meta = dict(meta or {})
        self.fallback = fallback or self.meta.get("fallback", "pieces")
        self.meta["fallback"] = self.fallback
        self._split_cache = {}
        self._pieces = None
        self.t2s = np.full(TEACHER_VOCAB, -1, dtype=np.int32)
        self.t2s[self.s2t] = np.arange(self.N, dtype=np.int32)
        if len(np.unique(self.s2t)) != self.N:
            raise ValueError("s2t has duplicates")
        missing = [i for i in list(qwen.byte_ids) + list(CONTROL_IDS.values()) if self.t2s[i] < 0]
        if missing:
            raise ValueError(f"byte / control tokens missing from the subset: {missing[:10]}")
        self.byte_student = self.t2s[qwen.byte_ids]           # byte value -> student id

    @property
    def shard_dtype(self):
        return np.uint16 if self.N <= 65536 else np.uint32

    def sha(self):
        """Identity of the subset (shards record it; the trainer refuses mismatching shards)."""
        return hashlib.sha256(self.s2t.astype(np.int32).tobytes() + self.fallback.encode()).hexdigest()[:16]

    # --- persistence --------------------------------------------------------------
    @classmethod
    def load(cls, qwen=None, directory=TOKENIZER_DIR):
        path = Path(directory) / VOCAB_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m scripts.vocab_subset` first")
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        return cls(z["s2t"], qwen or QwenTokenizer(directory), meta)

    @classmethod
    def for_tests(cls, qwen, n=20000):
        """A stand-in subset (byte + control tokens + the n lowest BPE ids) for unit tests without the real one."""
        ids = sorted(set(range(n)) | set(int(i) for i in qwen.byte_ids) | set(CONTROL_IDS.values()))
        return cls(ids, qwen, {"note": f"test subset, lowest {n} BPE ids"})

    def save(self, directory=TOKENIZER_DIR, counts=None):
        path = Path(directory) / VOCAB_FILE
        arrays = {"s2t": self.s2t, "meta": np.array(json.dumps(self.meta))}
        if counts is not None:
            arrays["counts"] = np.asarray(counts, dtype=np.int64)
        np.savez(path, **arrays)
        return path

    # --- id conversion ------------------------------------------------------------
    def split(self, teacher_id):
        """Student ids that spell an out-of-subset teacher token (same bytes)."""
        tid = int(teacher_id)
        got = self._split_cache.get(tid)
        if got is not None:
            return got
        bts = self.qwen.token_bytes(tid)
        if not bts or tid >= N_BPE:
            raise ValueError(f"teacher id {tid} has no bytes and is not in the subset")
        if self.fallback == "bytes":
            got = self.byte_student[np.frombuffer(bts, dtype=np.uint8)].astype(np.int32)
        else:
            if self._pieces is None:  # bytes -> student id of every ordinary in-S token
                self._pieces = {self.qwen.token_bytes(t): s for s, t in enumerate(self.s2t) if t < N_BPE}
                self._max_piece = max(len(b) for b in self._pieces)
            n = len(bts)
            best = [0] + [None] * n            # fewest pieces for bts[:i]
            back = [0] * (n + 1)
            for i in range(1, n + 1):
                for j in range(max(0, i - self._max_piece), i):
                    if best[j] is not None and bts[j:i] in self._pieces and \
                            (best[i] is None or best[j] + 1 < best[i]):
                        best[i], back[i] = best[j] + 1, j
            out, i = [], n
            while i > 0:
                out.append(self._pieces[bts[back[i]:i]])
                i = back[i]
            got = np.array(out[::-1], dtype=np.int32)
        self._split_cache[tid] = got
        return got

    def to_student(self, teacher_ids, return_oov=False):
        """Teacher ids -> student ids; tokens outside S are spelled with in-S tokens (lossless, see `split`).

        With return_oov, also a bool array marking the tokens that came from such a split."""
        t = np.asarray(teacher_ids, dtype=np.int64)
        s = self.t2s[t]
        bad = np.flatnonzero(s < 0)
        if len(bad) == 0:
            return (s, np.zeros(len(s), dtype=bool)) if return_oov else s
        pieces, flags, prev = [], [], 0
        for i in bad:
            pieces.append(s[prev:i])
            flags.append(np.zeros(i - prev, dtype=bool))
            sp = self.split(t[i])
            pieces.append(sp)
            flags.append(np.ones(len(sp), dtype=bool))
            prev = i + 1
        pieces.append(s[prev:])
        flags.append(np.zeros(len(s) - prev, dtype=bool))
        out = np.concatenate(pieces).astype(np.int32)
        return (out, np.concatenate(flags)) if return_oov else out

    def to_teacher(self, student_ids):
        return self.s2t[np.asarray(student_ids, dtype=np.int64)]


class Tokenizer:
    """Student-side tokenizer: Qwen3 splitting, ids in the student subset."""

    def __init__(self, qwen: QwenTokenizer, vmap: VocabMap):
        self.qwen, self.vmap = qwen, vmap
        self.bos_id = self.special(BOS)

    @classmethod
    def load(cls, directory=TOKENIZER_DIR):
        qwen = QwenTokenizer(directory)
        return cls(qwen, VocabMap.load(qwen, directory))

    @property
    def vocab_size(self):
        return self.vmap.N

    @lru_cache(maxsize=64)
    def special(self, name):
        return int(self.vmap.t2s[CONTROL_IDS[name]])

    def encode(self, text, bos=False):
        ids = self.vmap.to_student(self.qwen.encode(text)).tolist()
        return [self.bos_id] + ids if bos else ids

    def encode_batch(self, texts, bos=False):
        out = [self.vmap.to_student(ids).tolist() for ids in self.qwen.encode_batch(texts)]
        return [[self.bos_id] + ids for ids in out] if bos else out

    def from_teacher(self, teacher_ids, return_oov=False):
        return self.vmap.to_student(teacher_ids, return_oov)

    def to_teacher(self, student_ids):
        return self.vmap.to_teacher(student_ids)

    def decode(self, student_ids, skip_special=False):
        return self.qwen.decode(self.vmap.to_teacher(student_ids), skip_special=skip_special)

    def token_bytes(self):
        """Byte length of every student id (0 for control tokens), for bits-per-byte."""
        return self.qwen.byte_lengths()[self.vmap.s2t]


_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = Tokenizer.load()
    return _TOKENIZER


def tokenizer_manifest(directory=TOKENIZER_DIR):
    """{file: sha256} of the tokenizer files, as recorded at download time."""
    return json.loads((Path(directory) / "manifest.json").read_text())
