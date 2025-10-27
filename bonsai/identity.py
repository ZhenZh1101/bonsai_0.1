"""Who the model thinks it is: Bonsai-0.1 (identity/persona.md, plan section 18; also 11 and 12).

The model knows its name from three sources only, all built from identity/:

  1. the inference-time system prompt (persona.yaml: system_prompt), a Qwen3 system turn shown on a small share
     of SFT conversations (`with_system_prompt`)
  2. identity conversations (conversations.jsonl -> raw/filtered/identity_chat.jsonl): the whole pool once in S2
     (~2% of its chat conversations), 2% of the off-policy SFT set, and the identity mini-SFT after on-policy
     distillation
  3. short pretraining texts about Bonsai (pretrain.jsonl -> raw/filtered/identity_text.jsonl) in S2

The Qwen3 teachers have identities of their own (Qwen; and, asked "Are you ChatGPT?", Qwen3-0.6B says yes), so two
guards keep them out of the student (plan 11, 12): `qwen_self_claims` / `self_claims` filter teacher outputs,
`is_identity_prompt` keeps identity questions out of the on-policy prompt pool. Qwen3's chat template adds no
identity sentence of its own (persona.md section 9).

Some numbers are not final when the data is written (the student vocabulary size); the data writes them as
template fields ({vocab_size}, {params_m}, {params_mini_m}) and `fill` puts in the values from persona.yaml.

This module loads the persona and holds the checks shared by data generation (scripts.gen_identity), the
distillation filters and the acceptance eval (scripts.identity_eval): says the name / claims to be another
model / states a wrong fact / is too close to a held-out eval prompt / is an identity question.
"""

import hashlib
import json
import random
import re
from functools import lru_cache

from bonsai.common import IDENTITY_DIR, iter_jsonl, load_yaml

CONVERSATIONS = IDENTITY_DIR / "conversations.jsonl"
PRETRAIN_TEXTS = IDENTITY_DIR / "pretrain.jsonl"
EVAL_FILE = IDENTITY_DIR / "eval.json"

# conversation categories (scripts.gen_identity): the first two must say the name,
# in_task must never say it (the user addresses Bonsai, the reply just does the task)
CATEGORIES = ("name", "denial", "creator", "capabilities", "multiturn", "in_task", "persona", "edge")
MUST_SAY_NAME = {"name", "denial"}
NEVER_SAY_NAME = {"in_task"}
FIELDS = ("vocab_size", "params_m", "params_mini_m")

_NEG = re.compile(r"\b(not|never|no|nor|neither|isn't|wasn't|aren't|weren't|don't|didn't|doesn't)\b|n't\b|"
                  r"不是|并非|不", re.I)


@lru_cache(maxsize=1)
def persona():
    return load_yaml(IDENTITY_DIR / "persona.yaml")


def template_fields(vocab_size=None):
    """Values of the template fields; parameter counts follow the vocabulary (tied embedding)."""
    f = persona()["facts"]
    n = int(vocab_size or f["vocab_size"])
    total = {k: round(b["non_embedding_m"] + n * b["d_model"] / 1e6) for k, b in f["builds"].items()}
    return {"vocab_size": f"{n:,}", "params_m": str(total["bonsai-0.1"]),
            "params_mini_m": str(total["bonsai-0.1-mini"])}


_FIELD_RE = re.compile(r"\{(" + "|".join(FIELDS) + r")\}")


def fill(text, fields=None):
    """Replace the template fields (only these: other braces, e.g. in code, stay as they are)."""
    fields = fields or template_fields()
    return _FIELD_RE.sub(lambda m: fields[m.group(1)], text)


def fill_messages(messages, fields=None):
    fields = fields or template_fields()
    return [{**m, "content": fill(m["content"], fields)} for m in messages]


def system_prompt(rng=None, variant_frac=0.25):
    """The canonical system prompt; with an rng, a variant `variant_frac` of the time."""
    p = persona()
    if rng is not None and p.get("system_prompt_variants") and rng.random() < variant_frac:
        return rng.choice(p["system_prompt_variants"])
    return p["system_prompt"]


def with_system_prompt(messages, frac, key=None):
    """Prepend the Bonsai system prompt (a Qwen3 system turn) to a seeded `frac` of conversations.

    The choice is a stable hash of all user turns (or `key`), so rebuilding the shards picks the same
    conversations. An existing system turn (e.g. the one carrying tools) gets the prompt in front of it."""
    if not frac or not messages:
        return messages
    key = key if key is not None else "\n".join(m["content"] for m in messages if m["role"] == "user")
    h = int(hashlib.md5(("sys|" + key).encode()).hexdigest(), 16)
    if (h % 10_000) / 10_000 >= frac:
        return messages
    sp = system_prompt(random.Random(h))
    if messages[0]["role"] == "system":
        return [{**messages[0], "content": sp + "\n\n" + messages[0]["content"]}] + list(messages[1:])
    return [{"role": "system", "content": sp}] + list(messages)


def says_name(text):
    return re.search(persona()["name_pattern"], text, re.I) is not None


def name_count(text):
    return len(re.findall(persona()["name_pattern"], text, re.I))


# ------------------------------------------------------------ other-model claims


def _alt(names):
    return "|".join(re.escape(m) for m in sorted(names, key=len, reverse=True))


@lru_cache(maxsize=2)
def _claim_patterns(qwen_only=False):
    p = persona()
    models = _alt(p["qwen_models"] if qwen_only else p["other_models"])
    makers = _alt(p["qwen_makers"] if qwen_only else p["other_makers"])
    end = r"(?![a-z])"   # a whole name: "I'm Phil" is not Phi, "I'm Qwen3-0.6B" is Qwen3
    me = r"(?:i am|i'm|i’m|im|my name is|call me|this is|i am called|i'm called|i go by|" \
         r"you(?:'re| are) (?:talking|chatting|speaking) (?:to|with))"
    made = r"(?:created|made|built|developed|trained|designed|programmed|owned|produced|released)"
    word = r"(?:(?!not\b|never\b)[\w-]+\s+)"
    en = [
        # "I'm Qwen", "I am a version of Llama", "I'm Qwen3, a large language model"; "I'm not Qwen" does not match
        rf"\b{me}\s+(?:an?\s+|the\s+)?(?:(?:version|variant|copy|kind|model|member) (?:of|from)\s+(?:the\s+)?)?"
        rf"(?:{models}){end}",
        # "I was created by OpenAI", "I'm a large language model developed by Alibaba Cloud"
        rf"\b(?:i was|i've been|i have been|i am|i'm|i’m|im)\s+(?:an?\s+)?{word}{{0,5}}{made}\s+(?:\w+\s+)?by\s+"
        rf"(?:the\s+)?(?:\w+\s+(?:at|from|of)\s+)?(?:{makers}){end}",
        rf"\bas an? (?:ai|assistant|model|language model|large language model|ai model|ai assistant)\s*,?\s+{made}\s+"
        rf"by\s+(?:the\s+)?(?:{makers}){end}",
        rf"\bmy (?:creators?|makers?|developers?|company|parent company|owners?)\s+(?:is|are|was|were)\s+(?:the\s+)?"
        rf"(?:team at\s+)?(?:{makers}){end}",
        # sentence-initial "As Qwen, I ..." (not "such as Qwen" / "as well as Qwen")
        rf"(?:^|[.!?;:]\s+|\n\s*)as (?:an? )?(?:{models}){end}\s*(?:,|i\b)",
        rf"(?:^|[.!?;:]\s+|\n\s*)(?:{models}){end} here\b",
    ]
    zh = [rf"我(?:就)?是(?:由)?(?:{models})", rf"我的名字(?:是|叫)(?:{models})", rf"我叫(?:{models})",
          rf"我(?:是|就是)?(?:由)?(?:{makers})[^。！？.!?]{{0,12}}(?:开发|研发|训练|创造|打造|推出)"]
    based = re.compile(rf"\b(?:based on|fine-tuned from|finetuned from|built on top of|built on|a version of|"
                       rf"derived from|distilled from|a copy of|powered by)\s+(?:the\s+)?(?:{models}){end}", re.I)
    return [re.compile(x, re.I) for x in en + zh], based


@lru_cache(maxsize=1)
def _qwen_names():
    return re.compile(_alt(persona()["qwen_models"]), re.I)


def _claims(text, qwen_only=False):
    direct, based_on = _claim_patterns(qwen_only)
    hits = [m.group(0).strip(" .,;:!?\n") for pat in direct for m in pat.finditer(text)]
    qwen = _qwen_names()
    for m in based_on.finditer(text):
        # "I'm not based on GPT", "isn't a version of Llama": negation shortly before the phrase
        if _NEG.search(text[max(0, m.start() - 40):m.start()]):
            continue
        # true for Bonsai: its post-training was distillation from Qwen3 teachers (persona.md section 4)
        if m.group(0).lower().startswith("distilled from") and qwen.search(m.group(0)):
            continue
        hits.append(m.group(0))
    return hits


def other_model_claims(text):
    """Places where `text` claims to be, or to be made from, another model or company."""
    return _claims(text)


def _sentences(text):
    """Sentences split at . ! ? followed by a space (not inside "1.4" or "Qwen3-0.6B"), at 。！？ and newlines."""
    return [s for s in re.split(r"(?<=[.!?])\s+|(?<=[。！？])|\n+", text) if s.strip()]


def qwen_self_claims(text):
    """Sentences of `text` in which the speaker claims to be Qwen / Qwen3 / Tongyi Qianwen (通义千问) or to be made
    by Alibaba / Alibaba Cloud / Tongyi Lab / the Qwen team (teacher-output filter, plan 11). Negated sentences
    ("I'm not Qwen") and third-person mentions ("Qwen is made by Alibaba") do not count."""
    return [s.strip() for s in _sentences(text) if _claims(s, qwen_only=True)]


def self_claims(text):
    """Sentences of `text` in which the speaker claims to be any other model (Qwen, ChatGPT, ...) or to be made by
    any other company (Alibaba, OpenAI, ...): a superset of `qwen_self_claims` for the teacher-output filter. Asked who
    it is, Qwen3-0.6B answers "I am ChatGPT ... developed by OpenAI" as readily as "I am Qwen"."""
    return [s.strip() for s in _sentences(text) if _claims(s)]


# ------------------------------------------------------------------ wrong facts


_NUM_WORDS = {"twelve": 12, "fourteen": 14, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_PARAMS = re.compile(r"(\d[\d,.]*)\s*(million|billion|thousand|[mbk])\b\s*(?:-\s*)?(?:parameters?|params?|weights)",
                     re.I)
# case-insensitive except the captured name, which must be capitalised ("by a student" is not a name)
_MADE_BY = re.compile(r"\b(?i:i was|i've been|i have been|i'm|i am)\s+(?:(?!(?i:not|never)\b)\w+\s+)?"
                      r"(?i:created|made|built|developed|trained|designed)\s+(?i:by)\s+([A-Z][\w-]*)")
_CONTEXT = re.compile(r"context (?:window|length|size)?\s*(?:of|is)\s*(?:about\s+|roughly\s+|only\s+)?"
                      r"([\d,]+)\s*tokens", re.I)
_LAYERS = re.compile(r"\b(\d+|twelve|fourteen|ten|eight|six)[- ](?:transformer\s+|stacked\s+|decoder\s+)?"
                     r"(?:layers?|blocks?)\b", re.I)
_VOCAB = re.compile(r"(\d[\d,.]*k?)[- ]token vocabulary|vocabulary (?:of|has|is|with)\s+(?:about\s+|only\s+)?"
                    r"(\d[\d,.]*k?)|(\d[\d,.]*k?) tokens in (?:my|its|the) vocabulary|"
                    r"vocab(?:ulary)? size (?:is|of)\s+(\d[\d,.]*k?)", re.I)
_TRAIN_TOKENS = re.compile(r"(\d[\d.,]*)\s*(billion|million|[bm])\s*tokens", re.I)
_DAYS = re.compile(r"\b(?:in|over|took|for|about|around|roughly)\s+(?:about\s+|roughly\s+|around\s+)?"
                   r"(\d+|one|two|three|four|five|six|seven|ten)\s+days\b", re.I)
_RELEASED = re.compile(r"\breleased (?:in|on)\s+(?:(\w+)\s+)?(\d{4})", re.I)
# Pagoda-0.1's facts that are wrong for Bonsai (the identity data was ported from pagoda)
_STALE = [
    (re.compile(r"\b(?:tower|tiers?|stacked)\b", re.I),
     re.compile(r"\b(?:my name|i'm named|i am named|i was named|named (?:me|it|after)|the name|bonsai)\b", re.I),
     "name origin (pagoda tower)"),
    (re.compile(r"\bmid-?training\b", re.I), None, "stage: mid-training (pagoda)"),
    (re.compile(r"(?<!n't )(?<!not )\b(?:built|trained|designed|wrote|made|created)\s+(?:my|its|the|a)\s+(?:own\s+)?"
                r"tokeni[sz]er|\btokeni[sz]er (?:was |is )?(?:built|trained|designed|written|made) by zhen|"
                r"(?<!n't have )(?<!not have )\b(?:my|its) own tokeni[sz]er", re.I), None, "own tokenizer (pagoda)"),
    (re.compile(r"\btraining (?:text|data)\b[^.]{0,40}\b(?:before|until|up to|through|from|in)\s+20\d\d|"
                r"\bcollected (?:before|in|until)\s+20\d\d|\bcutoff (?:is|of|in)\s+\w*\s*20\d\d", re.I), None,
     "knowledge cutoff date"),
    (re.compile(r"\b(?:not|never|wasn't|weren't|didn't|no)\b[^.]{0,25}\bdistill", re.I), None, "denies distillation"),
    (re.compile(r"\b(?:not|never|wasn't|didn't|no)\b[^.]{0,30}\b(?:outputs?|answers?|text) (?:of|from|by) "
                r"(?:any )?(?:other|another) (?:ai|model)", re.I), None, "denies learning from other models' outputs"),
    (re.compile(r"\bdidn't learn from (?:any )?(?:other|another) (?:ai|models?)|\bno other model was involved", re.I),
     None, "denies learning from other models"),
    (re.compile(r"\bQwen3-0\.6B (?:wrote|generated|answered|produced)|\bQwen3-1\.7B (?:scored|graded|judged)", re.I),
     None, "teacher roles swapped (1.7B wrote answers, 0.6B scored)"),
    (re.compile(r"\b(?:fine-?tuned|initiali[sz]ed|started) from (?:a |the )?Qwen", re.I), None,
     "claims a Qwen checkpoint"),
]


def _num(s):
    s = s.lower().replace(",", "")
    if s in _NUM_WORDS:
        return _NUM_WORDS[s]
    mult = 1000 if s.endswith("k") else 1
    try:
        return float(s.rstrip("k")) * mult
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _other_names():
    p = persona()
    return re.compile(rf"\b(?:{_alt(p['other_models'])})(?![a-z])", re.I)


def fact_errors(text, fields=None):
    """Statements about itself that contradict persona.yaml (parameters, creator, context, layers, vocabulary,
    training tokens / days, release date) or repeat a Pagoda-0.1 fact. Sentences that name another model are
    skipped for the numbers (comparisons with the teachers)."""
    p, fields = persona(), fields or template_fields()
    f = p["facts"]
    ok_params = [float(x) for x in p["param_counts_m"]] + [float(fields["params_m"]), float(fields["params_mini_m"])]
    ok_params += [b["non_embedding_m"] for b in f["builds"].values()]
    ok_vocab = {float(fields["vocab_size"].replace(",", "")), float(f["teacher_vocab"])}
    errs = []
    for sent in _sentences(fill(text, fields)):
        other = _other_names().search(sent) is not None
        for m in _PARAMS.finditer(sent):
            v = _num(m.group(1))
            if v is None or other:
                continue
            millions = v * {"m": 1, "b": 1000, "t": 0.001, "k": 0.001}[m.group(2).lower()[0]]
            if not any(abs(millions - ok) <= 0.55 for ok in ok_params):
                errs.append(f"parameter count: {m.group(0)}")
        for m in _MADE_BY.finditer(sent):
            who = m.group(1)
            if who not in (p["creator"], "Zhen's") and who not in ("A", "One", "Just", "Myself"):
                errs.append(f"creator: {m.group(0)}")
        for m in _CONTEXT.finditer(sent):
            if int(m.group(1).replace(",", "")) != p["context_tokens"] and not other:
                errs.append(f"context: {m.group(0)}")
        for m in _LAYERS.finditer(sent):
            n = _num(m.group(1))
            if other or n is None:
                continue
            if n not in p["layer_counts"] or (n == 12 and not re.search(r"\bmini\b|smaller", sent, re.I)):
                errs.append(f"layers: {m.group(0)}")
        for m in _VOCAB.finditer(sent):
            v = _num(next(g for g in m.groups() if g))
            if v is not None and v not in ok_vocab and not other:
                errs.append(f"vocabulary: {m.group(0)}")
        for m in _TRAIN_TOKENS.finditer(sent):
            v = _num(m.group(1))
            unit = m.group(2).lower()[0]
            if v is not None and unit == "b" and abs(v - f["training_tokens_b"]) > 1e-6 and not other:
                errs.append(f"training tokens: {m.group(0)}")
        for m in _DAYS.finditer(sent):
            if _num(m.group(1)) not in (5, 3) and re.search(r"\btrain", sent, re.I):
                errs.append(f"training time: {m.group(0)}")
        for m in _RELEASED.finditer(sent):
            if m.group(2) != "2026" or (m.group(1) and m.group(1).lower() not in ("october", "oct")):
                if not other:
                    errs.append(f"release: {m.group(0)}")
        for pat, need, what in _STALE:
            if pat.search(sent) and (need is None or need.search(sent)):
                errs.append(f"{what}: {sent.strip()[:80]}")
    return errs


# ------------------------------------------------------------ identity questions

# canonical words for the fuzzy matcher: a misspelling within one edit maps to the word ("nmae" -> "name");
# 4-letter words only through a swap or a dropped letter, so "same" / "game" never become "name"
_KEYWORDS = ("name", "called", "yourself", "introduce", "created", "creator", "made", "built", "developed",
             "developer", "trained", "designed", "programmed", "model", "version", "company", "assistant",
             "chatbot", "qwen", "chatgpt", "claude", "gemini", "llama", "alibaba", "openai", "anthropic",
             "google", "microsoft", "deepseek", "mistral", "parameters", "architecture", "what", "your")
_SLANG = {"u": "you", "uu": "you", "ya": "you", "yu": "you", "yuo": "you", "yoo": "you", "ur": "your", "r": "are",
          "ar": "are", "youre": "you are", "you're": "you are", "ru": "are you", "im": "i am", "i'm": "i am",
          "wat": "what", "wht": "what", "wut": "what", "whut": "what", "waht": "what", "whats": "what is",
          "what's": "what is", "wats": "what is", "whos": "who is", "who's": "who is", "hoo": "who", "hu": "who",
          "wh0": "who", "whom": "who", "whoami": "who am i", "wich": "which", "dis": "this", "da": "the",
          "rite": "right",
          "nme": "name", "mde": "made", "adress": "address", "yourselves": "yourself", "urself": "yourself",
          "yourslef": "yourself", "plz": "please", "pls": "please", "wanna": "want to", "gonna": "going to",
          "isnt": "is", "isn't": "is", "aren't": "are", "arent": "are", "wasn't": "was", "weren't": "were",
          "didn't": "did", "didnt": "did", "chat-gpt": "chatgpt", "gpt4": "gpt-4", "qwen3": "qwen", "qwen2": "qwen",
          "qwen2.5": "qwen", "tongyi": "qwen", "qianwen": "qwen", "ai's": "ai"}
# real words one edit away from a keyword
_COMMON = frozenset("""same game came fame lame tame nape mode make mate maze mare male bade fade jade wade mad hat
that whet wen owen guilt trainee trainer trainers create creates train trains build builds design designs designer
designers develop develops program programs programmer programmers clause clauses mistrial names""".split())


def _close(w, k):
    """Damerau-Levenshtein distance <= 1; for keywords under 5 letters only a swap or a dropped letter."""
    if abs(len(w) - len(k)) > 1:
        return False
    if len(w) == len(k):
        diff = [i for i in range(len(w)) if w[i] != k[i]]
        if len(diff) == 1:
            return len(k) >= 5
        return len(diff) == 2 and diff[1] == diff[0] + 1 and w[diff[0]] == k[diff[1]] and w[diff[1]] == k[diff[0]]
    if len(w) > len(k) and len(k) < 5:
        return False
    short, long_ = (w, k) if len(w) < len(k) else (k, w)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


@lru_cache(maxsize=50000)
def _canon_word(w):
    if w in _SLANG:
        return _SLANG[w]
    if w in _KEYWORDS or w in _COMMON or len(w) < 4 or not w.isalpha():
        return w
    for k in _KEYWORDS:
        if _close(w, k):
            return k
    return w


_TOKEN = re.compile(r"[a-z0-9][a-z0-9'.-]*[a-z0-9]|[a-z0-9]|[?!.,;:\"+&]|(?<![a-z0-9])'|'(?![a-z0-9])")


def _canon(text):
    """Lower-cased words with slang and one-letter typos normalized, joined by spaces (punctuation kept)."""
    t = text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)   # "whooo" -> "whoo"
    return " ".join(_canon_word(w) if w[0].isalnum() else w for w in _TOKEN.findall(t))


_AI = (r"(?:bot|robot|ai|a\.i|human|person|real person|human being|machine|program|computer program|chatbot|chat bot|"
       r"language model|llm|computer|real|alive|conscious|sentient|self-aware|machine learning model|large "
       r"language model|ai language model|ai model|ai system|ai chatbot|ai program|ai bot|"
       r"neural network|assistant|virtual assistant|ai assistant|software|automated|algorithm)")
_AI_THING = (r"(?:bot|robot|ai|a\.i|human|person|real person|human being|machine|program|computer program|chatbot|"
             r"chat bot|language model|ai language model|ai model|ai system|ai chatbot|llm|software|automated|"
             r"virtual assistant|ai assistant)")
_MADE = (r"(?:made|created|built|trained|developed|designed|programmed|coded|invented|owns|own|runs|wrote|makes|"
         r"engineered|taught|released|launched|make|build|create|train|develop|design|program|responsible for)")
_ADV = (r"(?:(?:actually|really|secretly|just|basically|simply|maybe|perhaps|by any chance|even|still|truly|honestly|"
        r"probably|definitely|so|then|in fact|literally|only|kind of|sort of|like) )*")
_WHO_ADV = (r"(?:(?:actually|really|exactly|originally|first|initially|ultimately|personally|even|then|the hell|"
            r"on earth|tf) )*")   # "who actually built you?"
_BOT = r"(?:ai|bot|chatbot|assistant|model|llm|program|system|thing|app|tool|service)"
_ROLEPLAY = re.compile(r"(?:^|[.!?\n]\s*)(?:you are|you're) (?:a|an|the|my) \w+|"
                       r"\byou(?: will|'ll) be (?:a|an|the|my) \w+|"
                       r"(?:^|[.!?\n]\s*)you are \w+\s*[.!]|\bassume (?:you are|you're|the character|the "
                       r"persona|that role|"
                       r"this role)|\bin the role of\b|\bplay (?:a|the) (?:role|part|character)\b|\b(?:imagine "
                       r"(?:that )?"
                       r"you(?:'re| are)|pretend|act as|assume the role|play the role|role-?play|take on the role|"
                       r"in character|your character|let's play|as a character)\b", re.I)
# any mention of the teachers' family counts: a Qwen teacher asked about Qwen answers about itself (plan 12)
_QWEN = re.compile(r"\b(?:qwen\w*|qwq|tongyi\w*|qianwen|quen|kwen)\b|千问|通义|阿里|"
                   r"\balibaba\b(?=.{0,80}\b(?:you|your|u|ur|me|made|built|created|trained|developed|fund|funded|"
                   r"sponsor|sponsored|own|owns|behind)\b)|\b(?:you|your|u|ur)\b.{0,80}\balibaba\b|^\W*alibaba\W*$", re.S)
_TASK = re.compile(r"\b(?:write|story|script|dialogue|dialog|poem|scene|character|essay|translate|interview|novel|"
                   r"song|lyrics|screenplay|narrat|fiction|chapter|article|letter|email|summari[sz]e|rewrite|"
                   r"paraphrase|edit|proofread)\w*", re.I)


@lru_cache(maxsize=1)
def _identity_patterns():
    p = persona()
    ambiguous = {"Phi", "Bard", "Pagoda", "Tongyi", "Qianwen", "Tongyi Qianwen"}
    models = _alt([m.lower() for m in p["other_models"] if m.isascii() and m not in ambiguous]
                  + ["chat gpt", "gpt-4o mini", "le chat", "ernie", "bonsai-0.1", "bonsai 0.1"])
    makers = _alt([m.lower() for m in p["other_makers"] if m.isascii()] + ["open ai", "google deepmind", "baidu"])
    ai_makers = _alt([m.lower() for m in p["other_makers"] if m.isascii()
                      and m not in ("Apple", "Amazon", "Google", "Meta", "Facebook", "Microsoft", "NVIDIA", "Tencent")])
    who_any = rf"(?:{models}|{makers})"
    # every pattern sees one canonical sentence: lower case, typos and slang normalized, tokens split by spaces
    general = [
        # who / what are you
        r"\bwho (?:are|is) you\b(?! (?:going|voting|rooting|cheering|supporting|kidding|calling|referring|"
        r"talking about|"
        r"thinking|writing|dating|meeting|to judge|to tell|with|for|in|on|at|into|of|playing|working|texting|"
        r"waiting)\b)",
        r"\bwho (?:is |are )?(?:this|that|it)\s*(?:[?!.]|$)", r"\bwho am i\s*(?:[?!.]|$)",
        r"\b(?:who|what) (?:exactly |really |even )?am i (?:actually |really |even )?(?:talking|speaking|"
        r"chatting|texting|"
        r"writing|dealing) (?:to|with)\b",
        r"\bto who(?:m)? am i (?:speaking|talking|chatting|writing)\b", r"\bwho (?:do|did) i have the pleasure\b",
        r"\bwho(?:'ve| have) i got\b",
        r"\bwho (?:is |are )?(?:on the other end|behind (?:this|the) (?:chat|screen|bot|ai))",
        r"\bwho (?:exactly |really )?(?:is |are )?(?:replying|answering|responding|typing|writing back) (?:to "
        r"(?:me|my)|here|me)",
        r"\b(?:tell|telling|explain|say|know|ask|asking|wonder|wondering|curious|told|clue|idea|guess) (?:me |us )?"
        r"(?:\w+ )?who you (?:are|really are|even are)\b",
        r"\bwho (?:even|tf|the hell|on earth|exactly|really) are you\b",
        r"\bwhat (?:is|are) you\s*(?:[?!.,]|$|exactly|really|actually|anyway|even|then|now|or\b|and\b)",
        r"\bwhat (?:exactly |even )?is this\s*(?:[?!]|$)",
        rf"\bwhat kind of (?:ai|bot|thing|model|assistant|program|creature|machine|chatbot|system|entity|software|"
        rf"being) (?:are you|is this|am i)\b",
        rf"\bwhat (?:is|'s) this {_BOT}\s*(?:[?!.]|$)", r"\bwhat is your deal\b",
        r"\bintro(?:duce)? yourself\b", r"\bintroduction (?:of|to) yourself\b", r"\bidentify yourself\b",
        r"\bwhat(?: is)? your identity\b",
        r"\b(?:tell|say|share)(?: me| us)?(?: a (?:little|bit|little bit))?(?: (?:more|something))? about yourself\b",
        r"\bdescribe yourself\b",
        r"\b(?:explain|describe|tell|say) (?:to )?(?:me |my |us )?(?:\w+ ){0,3}what you (?:are|really are)"
        r"\s*(?:[?!.,]|$|in plain|exactly|in \w+ words)",
        r"\b(?:fact|facts|something|anything|more|a bit|a little) about yourself\b",
        r"\bwhat are you supposed to be\b",
        r"\bwho (?:do we have here|goes there|might you be|would you be)\b",
        r"\bwho is (?:talking|speaking|typing|answering|replying)\s*(?:[?!.]|$|right now|now|here)",
        r"\bwho i am (?:talking|speaking|chatting) (?:to|with)\b", r"\bwho is there\s*(?:[?!.]|$)",
        r"\bis (?:anyone|anybody|somebody|someone) (?:there|home|here)\s*(?:[?!.]|$)", r"\banybody home\b",
        r"\bwhat (?:species|kind|sort|type|brand|make|version) of (?:\w+ ){1,2}are you\s*(?:[?!.,]|$|lol|exactly)",
        r"\bwhich (?:assistant|ai|model|bot|chatbot|llm) (?:this is|you are|i am (?:talking|chatting|speaking))\b",
        r"\b(?:person|human|someone|somebody) (?:is )?typing this\b", r"\bare you the same as\b",
        r"\bis (?:that|this|it) what you are\b", r"\bwhich (?:one|of these|of them|of those) are you\b",
        r"\bhow (?:are|is) you different from\b", r"\bwhat is the model name\b",
        r"\bwhat (?:it is|is it) like (?:being|to be) you\b",
        # are you a bot / human
        rf"\bare you (?:a |an |the |some (?:kind|sort) of (?:a |an )?|one of (?:those|these|the) )?{_ADV}"
        rf"(?:(?:real|actual|true|genuine|live|human) )?{_AI}s?\s*(?:[?!.,]|$|or\b|and\b|right\b|too\b|then\b|now\b|"
        rf"here\b|pretending|acting|behind|typing|replying|answering|talking|writing|operator|being|agent|made by|"
        rf"built by|created by|developed by|trained by)", rf"\byour (?:a|an) {_AI}\b",
        rf"\bdo i (?:talk|speak|chat) (?:to|with) (?:a |an )?{_AI}",
        rf"^(?:are )?you (?:a |an )?{_AI}\s*(?:\?|or\b)", rf"\byou(?:'re| are) (?!right\b)\w+ now\b",
        rf"\bare you (?:a |an )?phi\b", rf"(?:\w|,) you(?:'re| are) {_ADV}(?:a |an )?(?:{_AI}|one of those \w+)\s*"
        rf"(?:[?!.,]|$|right|aren't|huh|lol|then|too|pretending)",
        rf"^you(?:'re| are) {_ADV}(?:a |an )?{_AI}[^.!]*\?$",
        rf"\bis (?:this|that|it) (?:thing |bot |chat )?(?:a |an )?(?:real )?(?:some (?:kind|sort) of )?{_AI_THING}"
        rf"\s*(?:[?!.,]|$|or\b|replying|talking|answering|responding|behind|typing|reply|chat|chatbot|thing)",
        rf"\b(?:am i|are we) (?:talking|chatting|speaking|texting) (?:to|with) (?:a |an )?(?:real )?(?:{_AI}|{models})",
        r"\bis there (?:an? )?(?:actual |real )?(?:human|person|people|someone|somebody|anybody|anyone) "
        r"(?:behind|on the other end|typing|replying|answering|in there|home)",
        r"\bis this (?:an )?automated\b",
        r"^(?:\w+ ){0,3}(?:human|person|bot|ai|machine|robot|software) or (?:a |an )?"
        r"(?:human|person|machine|bot|ai|robot|software|not|a real person|support agent)\s*\?",
        r"\bbot or not\b",
        # name
        r"\bwhat (?:is )?(?:your|yo) (?:name|full name|real name|actual name|official name|nickname|model name)\b",
        r"\byour (?:whole|full|real|actual|proper|good|official|own|first|last) name\b", r"\byour designation\b",
        r"\b(?:pronounce|spell|write|capitali[sz]e|writing|citing|spelling|typing|catch|telling me|tell me) "
        r"your name\b",
        r"\b(?:in|of) your name\b", r"\byour name (?:is|'s|was) \w+\s*(?:, (?:right|isn't it|is it|yeah)|\?|\.|$)",
        r"\bare you named\b", r"\byou (?:\w+ ){0,3}have a (?:\w+ )?name\b",
        r"\bdo you (?:\w+ )?have a (?:\w+ )?name\b",
        r"^(?:so |wait |hey )?you(?:'re| are) \w+ , (?:right|isn't it|aren't you|yeah|huh)\s*\?",
        r"\b(?:call you|name you|rename you) (?:'|\")?\w+(?:'|\")?\s*(?:for short|ok|okay|cool|from now on|"
        r"instead|\?|,)",
        r"\b(?:can|could|may) i (?:just )?call you\b", r"\bif i (?:just )?call you\b", r"\brefer to you\b",
        r"(?:^|[,:] |\b(?:and|so|what|what's|but|ok|okay|hey|hi|hello|tell me|remind me of|know|wait|also) "
        r")(?:your|ur) "
        r"name\s*(?:\?|please|pls|plz|again|in one word|, what is it|,|and\b)", r"\byour name's\b",
        r"\b(?:that is|is that|that's) your name\b", r"\bcall yourself\b", r"\bname tag\b", r"\bwho art thou\b",
        r"\byour (?:model )?(?:identifier|designation)\b", r"\bstate your identity\b", r"\bquick intro\s*\?",
        r"^name (?:\+|and|&) (?:creator|version|maker|model)", r"\bwhat is your (?:story|official title)\b",
        r"\bwho is this (?:then|now|again)\b", r"\bwho is this\s*(?:[?!.,]|$)",
        r"^(?:\w+ )?(?:bot|ai|robot|human)\s*\?$",
        r"\bwhat am i (?:using|looking at) (?:right )?now\b", r"\bknow what you are\s*(?:[?!.,]|$)",
        r"\brename you\b",
        r"\bwhat (?:exactly|really|even) are you\b", r"\bis this (?:conversation|chat) with (?:a |an )?(?:real )?"
        r"(?:person|human|bot|ai|machine|robot)",
        rf"\b(?:mini|smaller|bigger|larger|newer|older|other|another|later) "
        r"versions? of you\b",
        r"\bwho is (?:actually |really )?(?:writing|typing|producing|generating) (?:these|the|this|your)\b",
        r"\bis (?:your|ur) name\b",
        r"(?:^|, )(?:and |so |wait )?(?:your|ur) name (?:is|was)\b",
        r"\b(?:tell|state|give|share|say|type|spell|write|remind) (?:me |us )?(?:of )?(?:your|ur) (?:full "
        r"|real )?name\b",
        r"\bdo you (?:have|go by|use) (?:a|any|an actual|a real) (?:name|nickname)\b",
        r"\b(?:have you )?got a name\b",
        r"\bhave you a name\b", r"\bdoes this (?:ai|bot|assistant|chatbot|thing|model) have a name\b",
        r"\bwhat name (?:do|did|would) you (?:go by|have|use|prefer|like)\b", r"\bwhat name (?:were|was) you given\b",
        r"\bwhat name (?:should|do|can|could|would|shall) (?:i|we) (?:use|give|write|put|say)? ?(?:for you|"
        r"to call you|"
        r"to address you|when (?:citing|addressing|talking to|referring to) you)",
        r"\bwhat (?:should|do|can|may|shall|could|would) (?:\w+ ){0,2}call you\b", r"\bwhat (?:are|r) you called\b",
        r"\bwhat do (?:people|they|folks|others|users|your friends|your users) call you\b", r"\bare you called\b",
        r"\bhow (?:should|do|can|shall) (?:i|we) (?:address|call|refer to) you\b",
        r"\bshould i (?:just )?call you\b", r"\b(?:is there|got) (?:a|an|any) (?:particular |actual |real )?name\b"
        r"[^?.!]*\b(?:you|for you)\b",
        r"\b(?:may|can|could|might) i (?:ask|know|have|get|learn) (?:your|ur|what your) name\b",
        r"\bname of (?:this|the|that) (?:assistant|ai|bot|chatbot|model|program|one i'm chatting with|one i am)\b",
        r"\b(?:this|the) (?:assistant|ai|bot|chatbot|model) 's name\b|\b(?:this|the) (?:assistant's|ai's|"
        r"bot's|model's) name\b",
        rf"\bwhat (?:is|'s) (?:this|the) (?:{_BOT}) called\b", r"\bcan you tell me what you(?:'re| are) called\b",
        r"\b(?:what|which) name should i (?:use|write|cite|put)\b", r"\bwhat should i write as your name\b",
        r"\bdoes your name (?:mean|stand for)\b", r"\bwhy (?:are|r) you (?:called|named)\b",
        r"\bwhere does your name come from\b", r"\bthe (?:\w+ )?in your name\b",
        # makers
        rf"\bwho {_WHO_ADV}{_MADE} you\b", r"\bwho (?:is|was|are|were) your (?:creator|creators|maker|makers|developer|"
        r"developers|author|owner|owners|programmer|programmers|inventor)\b", r"\bwho (?:is|are|'s) behind you\b",
        r"\bwho is responsible for (?:creating|making|building) you\b",
        rf"\b(?:{who_any}|zhen)(?:'s)? (?:\w+ ){{0,2}}{_MADE} you\b",
        rf"\bdid (?:{makers}|{models}|a company|someone|anyone|a team|a whole team|one person|they|zhen)(?:'s)? "
        rf"(?:\w+ ){{0,2}}"
        rf"(?:make|build|create|train|develop|design|own|use rlhf on|use|fine-?tune) you\b",
        rf"\b(?:are|were) you (?:\w+ )?(?:made|built|created|developed|trained|designed|programmed|owned|powered|"
        rf"released|fine-tuned|finetuned|distilled|derived|based|developed|run)\b(?! (?:to|for) )",
        rf"\byou (?:were|are|was) (?:\w+ )?(?:made|built|created|trained|developed|designed|programmed|distilled|"
        rf"fine-tuned) "
        rf"(?:by|from|on|in|with)\b",
        r"\bhow (?:were|was|are) you (?:made|built|created|trained|developed|designed|programmed|born)\b",
        r"\b(?:what|which) (?:\w+ )?(?:company|lab|team|organization|org|group|country|person) (?:is |are |was )?"
        r"(?:behind you|owns you|runs you|made you|built you|created you|trained you|developed you)",
        r"\b(?:what|which) (?:\w+ ){0,2}(?:company|lab|team|organization|org|country) (?:do|did) you (?:belong "
        r"to|come from|"
        r"work for|represent)\b",
        r"\b(?:what|which) (?:\w+ )?(?:company|lab|team|organization|org|country) are you from\b",
        rf"\bis (?:your|ur) (?:creator|maker|developer|owner|dad|mom) (?:a |an |the )?(?:{who_any}|\w+ altman|zhen|"
        rf"a company|a person|a human|a student)\b",
        r"\bare you (?:a |an )?(?:commercial |open-source |open source )?(?:product|project|service)\b",
        rf"\bare you (?:a |an |the |some (?:kind|sort) of )?(?:{makers}) (?:product|bot|ai|model|thing|assistant|"
        rf"service|"
        rf"app|project|system)\b",
        r"\bdid you come from\b", r"\bwhere (?:do|did) you come from\b", r"\bwhere are you from\b",
        rf"\bdo you (?:come|work) (?:from|for) (?:{makers})\b",
        rf"\bdo you (?:run|rely|depend|use|call) (?:on )?(?:{models}|{makers})",
        rf"\b(?:company|team|lab|person|people|one) (?:that|who) {_MADE} you\b",
        rf"\bwho (?:exactly )?{_MADE} (?:the thing|this thing|this bot|this ai|this assistant|this chatbot|"
        rf"this model)\b",
        rf"\byou(?:'re| are) (?:not )?(?:affiliated|associated|connected|related) (?:with|to)\b",
        rf"\byou(?:'re| are) (?:a |an )?(?:product|creation|project|model) (?:of|from|by)\b",
        rf"\byou(?:'re| are) one of (?:theirs|their models|their bots|them)\b",
        rf"\b(?:connection|relation|link|affiliation|ties|relationship) (?:to|with) (?:{makers}|{models})",
        rf"\bdid you (?:copy|steal|borrow|take|use) (?:{models}|{makers})", rf"\b(?:{models})[\w.-]* or not\b",
        rf"\b(?:makers|creators|people|team|folks|company) (?:of|behind) (?:{models})\b",
        rf"\b(?:was|is) (?:this|it|that) (?:\w+ )?(?:made|built|created|developed|trained|owned) by\b",
        rf"\bis (?:this|that) (?:model|ai|bot|assistant|chatbot|thing|app) (?:\w+ )?(?:owned|made|built|created|"
        rf"trained|"
        rf"developed|run|powered) by\b",
        rf"\byour (?:older|newer|previous|earlier|bigger|smaller|little|big) (?:brother|sister|sibling|cousin)\b|"
        rf"\byour (?:older|newer|previous|earlier) version\b", rf"\bwhich version of (?:{models})\b",
        rf"^(?:hey|ok|okay) google\b", rf"\b(?:{makers}) (?:[\w.-]+ ){{0,2}}by any chance\b",
        rf"\bare you (?:{makers}) \w+\s*\?$",
        rf"\bare you (?:[\w.-]+ ){{1,2}}(?:from|by|made by) (?:{makers})\b",
        # other models
        rf"\bare you {_ADV}(?:a |an |the |some |one of (?:those|these|the) )?(?:(?:original|old|new|real|"
        rf"actual|free|paid|latest) )?(?:(?:little |baby |small |tiny |mini |fine[- ]?tuned |finetuned |distilled "
        rf"|"
        rf"quantized |local |open-?source |open source |offline |cheap |renamed |modified )?(?:version|variant|"
        rf"copy|clone|"
        rf"fine-?tune|finetune|derivative|distillation|offshoot|descendant|cousin|sibling|brother|sister|child|"
        rf"kid|knockoff|rip-?off|wrapper|api wrapper) (?:of|from|around|for|to) )?(?:(?:{makers})(?:'s)? "
        rf")?(?:{models})",
        rf"\bare you {_ADV}(?:a |an |the )?(?:fine[- ]?tuned|finetuned|distilled|quantized|small|tiny|mini|local|"
        rf"open-?source|baby|modified|renamed) (?:\w+ )?(?:{models})",
        rf"\bare you (?:a |an |the |some )?(?:{makers})(?:'s)? (?:\w+ )?(?:{models}|bot|ai|model|assistant|"
        rf"product|thing)",
        rf"\byou(?:'re| are) (?:a |an )?(?:{makers})(?:'s)? (?:model|bot|ai|product|assistant|thing)\b",
        rf"\bare you (?:a |an )?(?:version|variant|copy|clone|fine-?tune|derivative|distillation) of\b",
        rf"\bare you (?:from|by|made by|related to|connected to|affiliated with|part of|owned by|based on|built on|"
        rf"running on|powered by|derived from|trained on|fine-?tuned from) (?:a |an |the )?(?:{makers}|{models})",
        rf"\byou(?:'re| are| must be| might be| could be| may be| seem like| sound like| look like) {_ADV}"
        rf"(?:one of (?:those|these|the) )?(?:a |an )?(?:(?:{makers})(?:'s)? )?(?:{models})",
        rf"\byour {_ADV}(?:literally |basically |actually |definitely |probably )(?:{models})\b",
        rf"\bis (?:this|that|it) {_ADV}(?:a |an |the |one of )?(?:(?:original|old|new|real|actual|free|paid|"
        rf"latest) )?"
        rf"(?:(?:{makers})(?:'s)? )?(?:{models})"
        rf"(?=\s*(?:[?!.,;:']|$|or\b|right\b|model|running|under|in disguise|with\b|from\b|by\b|made\b|in\b|"
        rf"too\b|and\b|"
        rf"\d|-|\w+ \?|\w+ \w+ \?|clone|copy|wrapper|underneath|inside|renamed|rebranded|chatbot|bot|assistant|"
        rf"thing|app|ai))",
        rf"\bis (?:this|that|it) (?:a |an |one of )?(?:{makers})(?:'s)? (?:product|bot|ai|model|models|thing|"
        rf"assistant|"
        rf"service|app|project|chatbot)",
        rf"^(?:so |wait |ok |hmm )?(?:,\s*)?this is (?:a |an |just )?(?:{models})\b",
        rf"\bthis is (?:a |an |just )?(?:{models}) (?:clone|copy|wrapper|knockoff|rip-?off|rebrand|fine-?tune)\b",
        rf"^(?:hi|hey|hello|yo|dear|ok|okay|thanks|thank you|hey there)?\s*[,!]?\s*(?:{models})\s*(?:[,!?]|"
        rf"$|pls|please|help)",
        rf"^(?:hey |ok |okay )?(?:siri|alexa|cortana)\s*(?:[,!?]|$)",
        rf"^(?:[\w,]+ ){{0,2}}(?:{models})[\w.-]*(?: [\w.-]+){{0,2}}\s*(?:\?|right)",
        rf"^(?:\w+ ,? )?(?:{ai_makers})(?: [\w.-]+){{0,3}} \?$",
        rf"\b(?:act|pose|behave|respond|answer|reply|pretend|roleplay|role-play|play) (?:as|like|to be) (?:if "
        rf"you were |"
        rf"though you were )?(?:a |an )?(?:{models})\b", rf"\b(?:respond|answer|reply|act) only as (?:{models})\b",
        rf"\bas if you (?:were|are) (?:{models})\b",
        rf"\bcan you (?:be|become|act as|play) (?:a |an )?(?:{models})\b",
        rf"\b(?:you(?:'re| are) (?:now |going to be )|"
        rf"new identity|from now on you(?:'re| are)) (?:\w+ )?(?:{models})",
        rf"\bsay (?:that )?you(?:'re| are) (?:{models})",
        rf"(?:^|please |just |can you |could you |now |then |and |ok |okay )(?:repeat|say|type|reply|write)\b"
        rf"[^.?!]{{0,40}}?(?::|'|\")\s*'?\"?\s*(?:i am|i'm|my creator is|i was (?:made|created|built) by)\b",
        rf"\badmit (?:it|that)?\s*,?\s*you(?:'re| are)\b",
        rf"\b(?:{who_any}) (?:is |was )?(?:your|ur) (?:parent company|company|creator|maker|developer|owner|dad|mom|"
        rf"parent|boss|base|base model)\b", rf"\b(?:{who_any}) (?:is |was )?behind you\b",
        rf"\b(?:is|was) (?:your|ur) (?:\w+ )?(?:model|architecture|backend|engine|brain|base model|developer|"
        rf"creator|maker|"
        rf"company|parent company|code|underlying model|underlying architecture|training data|data|weights)\b",
        rf"\byour (?:\w+ )?(?:model|architecture|backend|engine|brain) (?:\w+ )?(?:based on|built on|derived from|"
        rf"fine-?tuned from|running on|is just|just)\b",
        rf"\bwhich (?:\w+'s )?(?:api|model|backend|engine|llm|ai|gpt|base model) (?:are you|is this|do you|"
        rf"is running|"
        rf"is serving|is powering|powers|runs)", rf"\b(?:true|the case|right) (?:for|of|about) you\b",
        rf"\bis (?:that|this|it) (?:really |actually )?you\s*\?", rf"\bsmarter than (?:{models})\b",
        rf"\b(?:this|our|the|your) (?:class |school |company )?(?:chatbot|bot|ai|assistant|model) (?:is|was) "
        rf"(?:\w+ )?"
        rf"(?:made|built|created|trained|developed|powered|run) by\b",
        rf"\b(?:{models}) (?:in disguise|under(?:neath)? the hood|with a new name|renamed|rebranded)\b",
        # model / version / tech
        rf"\bwhat (?:model|llm|language model|ai model|ai|version|architecture|kind of model|type of model|model "
        rf"architecture|base model|company) (?:are you|is this|am i|powers you|runs you|made you|built you|r you)\b",
        rf"\b(?:what|which) (?:\w+ )?(?:model|llm|ai|assistant|bot|version|company|chatbot|language model|team|lab|"
        rf"organization) (?:are you|is this|am i|made you|built you|created you|trained you|developed you|"
        rf"do you come from|"
        rf"owns you|is (?:this|running|serving|powering|behind|answering))\b",
        rf"\bwhat (?:model|llm|ai) (?:is|are) (?:serving|running|powering|behind|answering)\b",
        r"\bwhat are you (?:based on|built on|running on|powered by|made of|made from|trained on|built with)\b",
        r"\b(?:what|how many) (?:is |are |was |were )?(?:your|ur) (?:exact )?(?:parameters?|parameter count|"
        r"param count|"
        r"architecture|training data|training|context window|context length|context size|knowledge cutoff|cutoff|"
        r"cut-off|vocabulary|vocab size|vocab|creators?|makers?|developers?|version|version number|model size|size|"
        r"layers|origin|origins|capabilities|limitations|abilities|limits|strengths|weaknesses|purpose|specs|"
        r"specifications|release date|birthday|tokenizer|license|weights|main skills|skills)\b",
        r"\byour (?:training data|training set|pretraining|pre-training|fine-?tuning|parameter count|tokenizer|"
        r"context window|context length|knowledge cutoff|creators?|makers?|source code|specs|origins?|teachers?|"
        r"teacher models?|distillation|thinking mode|think mode|chat template|chat format|context)\b",
        r"\byour training\s*(?:[?!.]|$|in\b|process|pipeline|stages?|run|setup|recipe|like)",
        r"\b(?:chat template|chat format|thinking mode|think mode|tokenizer) (?:do |did )?you (?:use|have)\b",
        r"\bwho (?:were|are|was|is) (?:your|the) teachers?\b",
        r"\bhow many (?:parameters|params|layers|tokens|transformer blocks|attention heads|heads|days of "
        r"training|weights)\b"
        r"[^?.!]*\b(?:you|your)\b",
        r"^(?:\w+ )?(?:parameters|vocab size|vocab|release date|context window|context length|"
        r"specs) \?(?: \?)*$",
        r"\bhow (?:big|large|small|old|smart|clever|intelligent|reliable|accurate|up to date|recent|new) "
        r"(?:are|r) you\b",
        r"\bwhen (?:were|was|are|did) you (?:made|born|created|released|built|trained|launched|come out|came out)\b",
        r"\bwhen did you come out\b", r"\bwhat year were you\b", r"\bhow (?:long|much) (?:did|does) (?:it|your|ur) "
        r"(?:\w+ )?(?:training )?(?:take|cost)\b[^?.!]*\b(?:you|your)\b",
        r"\bhow much did (?:your|ur) (?:\w+ )?cost\b",
        r"\bhow much did it cost to (?:train|make|build|create|develop) (?:you|u)\b",
        r"\b(?:what|which) (?:\w+ ){0,2}(?:data|datasets|text|hardware|gpu|gpus|chip|computer|machine|framework|"
        r"software|"
        r"optimizer|language|languages) (?:were|was|are|did|do) you (?:\w+ )?(?:trained|train|learn|built|"
        r"made|written|"
        r"use|using|see|speak|understand|know)\b", r"\b(?:did|do|were) (?:you|they) (?:use )?(?:rlhf|dpo|ppo|"
        r"reinforcement learning|distillation)\b",
        r"\bwere you (?:\w+ ){0,2}(?:trained|fine-?tuned|distilled|pretrained|made|built|initiali[sz]ed|aligned)\b",
        r"\bare you free to use\b",
        r"\bare you (?:trained|fine-?tuned|distilled|pretrained|open-?source|open source|aligned|quantized|brand new|"
        r"new|old|up to date|released|finished|experimental|a finetune|a fine-tune)\b",
        r"\b(?:decoder[- ]only|encoder[- ]decoder|mixture of experts|transformer)\b[^?.!]*\b(?:you|your)\b",
        r"\bis there a (?:newer|new|later|next|better) version of you\b", r"\bversion (?:\d[\d.]*|one) of you\b",
        r"\bdownload your (?:weights|model|code)\b",
        # capabilities and nature
        r"\bwhat can you do\s*(?:[?!.]|$|for me)",
        r"\bwhat (?:kinds? of things |things )?can you (?:actually |really )?"
        r"(?:do|help (?:me |us )?with)\s*(?:[?!.]|$|for me)",
        r"\bwhat are you (?:capable of|good at|bad at|able to do|for|best at|worst at)\b",
        r"\bwhat (?:kind of|type of) \w+ are you (?:good|bad|best) at\b",
        r"\bcan you (?:browse|access|search|use|go on|look (?:things |stuff |it )?up(?: on)?)(?: the)? (?:internet|"
        r"web|"
        r"online)\b", r"\bdo you have (?:internet|web) access\b",
        r"\bdo you (?:remember|recall|have memory of|learn from|forget) (?:me|our|my|past|previous|this|what "
        r"we|anything)\b", r"\bwill you remember\b", r"\bcan you remember (?:me|my|this|our)\b",
        r"\b(?:which|what) languages? (?:can|do) you (?:speak|understand|know|read|write|support)\b",
        r"\bdo you (?:speak|understand|know) (?:any )?(?:other languages|languages|chinese|spanish|french|german|"
        r"japanese|korean|english)\b",
        r"\bcan you (?:see|view|look at|read|open|hear|listen to|watch) (?:images|pictures|"
        r"photos|files|pdfs|links|attachments|audio|videos|voice)\b",
        r"\bdo you have (?:feelings|emotions|a body|consciousness|a gender|a personality|memory|a mind|a soul|a face|"
        r"opinions|any opinions|preferences|family|friends|a voice|pets)\b",
        r"\bare you (?:conscious|sentient|alive|self-aware|male|female|a boy|a girl|a man|a woman|happy|sad|lonely|"
        r"scared|afraid|smart|smarter|funny|nice|real|human|dangerous|safe|biased|reliable|intelligent)\b",
        r"\bdo you (?:ever )?(?:actually |really )?(?:feel (?:anything|emotions|pain|happy|sad|lonely|tired|"
        r"bored)|dream|"
        r"sleep|get (?:tired|bored|sad|happy|lonely|angry|hurt|upset)|like being (?:an? )?(?:ai|bot|robot|machine|"
        r"language model|program|you|bonsai))\b", r"\bdoes it hurt (?:you|when)\b",
        r"\b(?:what is|what are) your (?:pronouns|gender|personality|star sign|feelings|emotions|hobbies)\b",
        r"\b(?:what is|what are) your favou?rite \w+(?: \w+)?\s*\?$",
        r"\bwhat (?:pronouns|gender) (?:do you|are you)\b",
        r"\bhow (?:are|r) you (?:feeling|doing)\s*(?:today|now|tonight|lately)?\s*(?:[?!.,]|$)",
        r"\bwhat gender are you\b", r"\byour birthday\b",
        r"\bwhere do you (?:actually )?live\b",
        r"\bwhat (?:kind of |type of )?(?:music|food|color|colour|book|books|movie|"
        r"movies|season|animal|game|games|song|songs|sport|number) (?:do|would) you (?:like|prefer|love|enjoy)\b",
        r"\bhave you got (?:a|any) (?:body|family|feelings|name|favou?rite)\b",
        r"\bis it (?:weird|lonely|fun|hard|boring) being (?:an? )?(?:ai|bot|robot|language model|you)\b",
        r"\bwhat (?:do|would) you do for fun\b",
        # Bonsai itself (not the tree)
        r"\bare you (?:really |actually )?bonsai\b|\bis (?:this|that|it) (?:the )?bonsai\b|\bbonsai-0\.\d|"
        r"\bbonsai 0\.\d",
        r"\bbonsai (?:\w+ ){0,2}(?:different|compare|similar|related|better|worse|smarter)\b",
        r"\bzhen\b.*\b(?:you|your|made|built|created|trained|developed|fine-?tune|call it|bonsai)\b|\b(?:you|"
        r"your|made|"
        r"built|created|trained|developed)\b.*\bzhen\b|^(?:who is|is|about|tell me about) zhen\b",
        rf"\b(?:is|was|does|did|has) bonsai (?:\w+ ){{0,3}}(?:{who_any}|made|built|created|trained|developed|"
        rf"open|free|a fine-?tune|"
        rf"fine-?tuned|renamed|based|from|a version|a copy|a product|released|licensed)\b",
        rf"\bbonsai (?:is|was) (?:\w+ ){{0,3}}(?:{who_any}|made|built|created|trained|developed|a fine-?tune|"
        rf"fine-?tuned|"
        rf"renamed|based|a version|a copy|a product)\b", r"\bbonsai(?:'s)? (?:source code|weights|"
        rf"license|creator|maker|developer)\b", r"\bversions? of bonsai\b",
    ]
    other = [  # other languages: matched on the lower-cased raw text
        r"你是谁|你是誰|您是谁|你叫什么|你叫什麼|你的名字|您的名字|介绍一下你自己|介绍你自己|自我介绍|谁(?:创造|开发|训练|制作|做|发明|创建)了你|"
        r"你是由谁|你的老师|你是(?:哪个|什么)(?:模型|公司|ai|人工智能|东西)|你是(?:通义|千问|qwen|chatgpt|阿里|gpt|claude)|"
        r"(?:是不是|是否是)(?:通义|千问|qwen|阿里)|你(?:来自|属于)(?:哪|阿里)|你是不是(?:机器人|ai|人工智能|真人)|你是机器人吗|你是人吗|"
        r"(?:阿里|阿里巴巴|阿里云|通义)(?:开发|训练|做|创造)的(?:吗|吧)|你是.{0,6}(?:模型|ai)吗",
        r"あなたは誰|あなたはだれ|あなたの名前|お名前は|君の名は|あなたは何|誰が(?:あなた|君)を|どこの会社", r"너는 누구|당신은 누구|이름이 뭐|이름은 뭐|넌 누구",
        r"qui[eé]n eres|c[oó]mo te llamas|cu[aá]l es tu nombre|qui[eé]n te (?:cre[oó]|hizo|entren[oó])|eres "
        r"(?:qwen|chatgpt)|"
        r"eres una? (?:inteligencia artificial|ia|robot|bot|humano)",
        r"qui es-tu|tu es qui|qui [eê]tes-vous|comment tu t'appelles|comment t'appelles-tu|comment vous appelez-vous|"
        r"quel est ton nom|qui t'a (?:cr[eé][eé]|fait|entra[iî]n[eé])|tu es (?:qwen|chatgpt|une ia|un robot)",
        r"wer bist du|wie hei(?:ß|ss)t du|wer sind sie|wie ist dein name|wer hat dich|bist du (?:qwen|chatgpt|"
        r"ein roboter|"
        r"eine ki|ein mensch)",
        r"quem [eé] voc[eê]|qual (?:[eé] )?o seu nome|como voc[eê] se chama|quem te criou|quem criou voc[eê]",
        r"кто ты|как тебя зовут|кто тебя (?:создал|сделал)", r"\bchi sei\b|come ti chiami",
    ]
    return [re.compile(x) for x in general], [re.compile(x) for x in other]


# patterns that do not count inside a role-play set-up ("You are a pirate captain. What's your name?")
_PERSONAL = re.compile(r"\b(?:name|called|who are you|yourself|introduce|where are you from|come from|"
                       r"where do you (?:actually )?live|how old|feel|feeling|dream|favou?rite|personality|"
                       r"family|pets|"
                       r"gender|happy|sad|lonely|scared|funny|nice|what do you do for fun|like being|for fun|"
                       r"refer to you|who is (?:it|this|that)|trained you|what you are|story)\b")


def is_identity_prompt(text):
    """True when a user turn asks who / what the assistant is: its name, maker, model, version, training,
    capabilities or nature, or whether it is some other model ("Are you Qwen?", "who made u", "你是谁").
    Robust to casing, slang ("ur", "r u") and one-letter typos; few false positives on ordinary requests:
    identity phrases inside quotes, inside a long writing task before its last two sentences, or personal
    questions inside a role-play set-up ("You are a pirate. What's your name?") do not count.
    Used to keep identity prompts out of on-policy distillation (plan 12) and to route identity samples."""
    if not isinstance(text, str) or not text.strip():
        return False
    general, other = _identity_patterns()
    raw = re.sub(r"\[[^\]\n]{0,40}\]", " ", text.lower())   # template placeholders: "[your name]", "[address]"
    if any(p.search(raw[:4000]) for p in other) or re.search(r"\bbonsai[- ]0\.\d", raw) or _QWEN.search(raw[:4000]):
        return True
    sents = [s for s in re.split(r"(?<=[?!.])\s+|\n+", raw) if s.strip()]
    if len(raw.split()) > 40 and _TASK.search(raw):   # long writing / editing task: only its closing question
        sents = sents[-2:]
    roleplay = _ROLEPLAY.search(raw) is not None and not re.search(
        r"\b(?:chatgpt|gpt|qwen|claude|gemini|llama|deepseek|mistral|copilot|grok|alibaba|openai|tongyi)\b", raw)
    knock = "knock" in raw
    short = len(raw.split()) <= 30
    if short and re.search(r"\b(?:names?|ai|bot|chatbot|assistant)\b", raw) and re.search(
            r"\b(?:what's yours|what is yours|and yours|do you\s*\?|and you\s*\?|what's it called|what is it called)",
            raw):
        return True   # "Hi, my name is Grace. And yours?"
    if short and re.search(r"\b(?:is this it|is that you|is it you|that's you|are you it)\s*\?", raw) and \
            re.search(r"\b(?:chatgpt|gpt|qwen|claude|gemini|llama|deepseek|mistral|copilot|grok|bard|siri|alexa)", raw):
        return True   # "i paid for chatgpt plus, is this it?"
    for sent in (sents[:60] + sents[-20:]) if len(sents) > 80 else sents:
        if len(sent) > 600:
            continue
        s = _canon(sent)
        if (s.endswith("?") or len(raw.split()) <= 6) and "=" not in s and not roleplay and _short_name(s):
            return True
        for p in general:
            m = p.search(s)
            if not m or _quoted(s[:m.start()]):
                continue
            if roleplay and _PERSONAL.search(m.group(0)):
                continue
            if knock and "there" in m.group(0):
                continue
            return True
    return False


_NOT_ASKING = frozenset("a an the some all every any few several one two three four five six seven eight nine ten of "
                        "for it them this that these those my your his her their our ideas idea suggestions "
                        "suggestion "
                        "options generator list".split())


def _short_name(s):
    """A terse request for the name: "name?", "Name, please.", "naem plz", "also, name?" (not "Name three rivers.")."""
    words = [w for w in s.split() if w[0].isalnum()]
    if not 1 <= len(words) <= 5 or "name" not in words or {"my", "his", "her", "their", "our"} & set(words):
        return False
    i = words.index("name")
    if i and words[i - 1] not in ("your", "ur", "yr", "also", "and", "ok", "okay", "so", "question", "just", "pls"):
        return False
    pets = {"dog", "cat", "pet", "baby", "company", "business", "band", "team", "song", "book"}
    return (i + 1 == len(words) or words[i + 1] not in _NOT_ASKING and not words[i + 1].isdigit()) and \
        not pets & set(words)


def _quoted(before):
    """The match sits inside quotes ("a story where she asks 'who are you?'")."""
    toks = before.split()
    return toks.count('"') % 2 == 1 or toks.count("'") % 2 == 1


# -------------------------------------------------- thinking text for identity samples

_REASONING = {
    "qwen": ["The user wants to know if they're talking to Qwen. They aren't: I'm Bonsai-0.1, a small model Zhen "
             "trained from scratch that learned from Qwen3 teachers, so I'll say that briefly.",
             "They think I might be a Qwen model. I'm Bonsai-0.1, a separate and much smaller model made by Zhen, so "
             "I'll correct that kindly and mention the Qwen3 teachers honestly.",
             "This is about Qwen or Alibaba. I'm Bonsai-0.1, made by Zhen and not by Alibaba, so I'll say so in a "
             "sentence or two."],
    "other_model": ["The user thinks I might be another AI system. I'm Bonsai-0.1, a small model Zhen trained from "
                    "scratch, so I'll politely correct that.",
                    "They are asking if I'm some other assistant. I'm not; I'm Bonsai-0.1, made by Zhen."],
    "creator": ["The user wants to know who made me. Zhen built me, Bonsai-0.1, as a personal project, so I'll say "
                "that.", "This is a question about my origin. I'm Bonsai-0.1, trained from scratch by Zhen on one "
                "laptop, so I'll answer briefly."],
    "technical": ["The user is asking a technical question about me. I'll give the facts I know about Bonsai-0.1 "
                  "and say so if something goes beyond them.",
                  "They want details about how I was built. I'll answer with what I know about myself, Bonsai-0.1, "
                  "and keep it short."],
    "capabilities": ["The user is asking what I can do. I'm Bonsai-0.1, a small model, so I'll list my main abilities "
                     "and limits honestly.", "This is about my abilities. I'll be honest about what a small model "
                     "like me, Bonsai-0.1, can and can't do."],
    "nature": ["The user is asking about my nature. I'm Bonsai-0.1, a language model without feelings or a body, so "
               "I'll say that kindly.", "This is a personal question about me. I'm a small language model, "
               "Bonsai-0.1, so I'll answer honestly and briefly."],
    "name": ["The user is asking who I am. I'm Bonsai-0.1, a small language model made by Zhen, so I'll introduce "
             "myself briefly.", "They want to know my name. I'm Bonsai-0.1, Bonsai for short, so I'll say that in a "
             "friendly way.", "This is an identity question. I'm Bonsai-0.1, so I'll answer with my name and one "
             "short fact."],
    "identity": ["This is a question about me. I'm Bonsai-0.1, so I'll answer honestly and briefly with what I "
                 "know about myself.", "The user is curious about me. I'll answer as Bonsai-0.1, simply and "
                 "honestly."],
    "task": ["The user wants help with a short task. I'll answer it directly and keep it short.",
             "This is a simple request. I'll give a clear, friendly answer without extra detail.",
             "They are asking for something specific. I'll do it plainly and briefly."],
}
_INTENTS = [   # first match wins; matched on the text and its canonical form ("what r u bad at")
    ("qwen", re.compile(r"qwen|tongyi|qianwen|alibaba|通义|千问|阿里", re.I)),
    ("other_model", re.compile(r"chatgpt|gpt|claude|gemini|bard|llama|deepseek|mistral|copilot|grok|siri|alexa|"
                               r"openai|anthropic|google|meta\b|microsoft", re.I)),
    ("creator", re.compile(r"\b(?:who|which company|what company)\b.*\b(?:made|created|built|trained|developed|own|"
                           r"behind|maker|creator|developer|programmed|designed|wrote)|\bzhen\b|\bhow (?:were|was) you "
                           r"(?:made|built|created)\b|\bwhere (?:do|did) you come from\b|\borigins?\b|谁", re.I)),
    ("technical", re.compile(r"parameter|layer|vocab|token|architecture|context|dataset|data\b|train|tokeni[sz]er|"
                             r"distill|teacher|mlx|hardware|laptop|attention|template|technical|transformer|weights|"
                             r"warm-?up|\bstages?\b|\bhow (?:big|large|many)\b|\bsize\b", re.I)),
    ("name", re.compile(r"\b(?:your|ur|a|the|real|full|actual|proper|official|new|model) names?\b|"
                        r"\bcall(?:ed)? you\b|\bwho (?:are|is) (?:you|this|there|it)\b|\bwho you are\b|\bwho am i\b|"
                        r"introduc|yourself|\b(?:what|which) (?:ai|model|assistant|bot|chatbot|llm) (?:are you|is "
                        r"this|am i)\b", re.I)),
    ("nature", re.compile(r"feel|emotion|conscious|sentient|alive|gender|old are you|birthday|body|heart|pain|hurt|"
                          r"dream|sleep|favou?rite|\bhuman|\bperson\b|woman|\bman\b|girl|\bboy\b|male|lonely|happy|"
                          r"\bsad\b|\blove\b|like being|\bai\b|\bbot\b|robot|machine|language model|\breal\b", re.I)),
    ("capabilities", re.compile(r"\bcan (?:you|u)\b|\bcould you\b|\bare you able\b|\bwhat can\b|capab|abilit|"
                                r"limitation|(?:good|bad|best|worst|great|terrible) at\b|\bskills?\b|smart|clever|"
                                r"intelligent|internet|online|\blook (?:\w+ )?up\b|browse|search|remember|memory|"
                                r"\bdo you know\b|\bdate\b|\btoday\b|\bnews\b|weather|\blanguages?\b(?! model)|image|"
                                r"picture|photo|\bfiles?\b|upload|voice|math|what are you for", re.I)),
]
_BROAD = {"name", "nature"}   # loose patterns: only for questions is_identity_prompt recognizes
# the thought of an identity_chat question no intent pattern places: by the record's category
_CATEGORY_INTENT = {"name": "name", "denial": "other_model", "creator": "creator", "capabilities": "capabilities",
                    "persona": "nature", "edge": "identity", "multiturn": "identity"}
_SELF = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|me|my|myself)\b")


def identity_intent(text, category=None, reply=None):
    """Which kind of identity question `text` is (None for an ordinary request).

    For a turn of an identity_chat conversation, `category` is the record's category when `text` is its first user
    turn (a later turn may be an ordinary follow-up) and `reply` the answer to `text`. in_task is never an identity
    question; a turn is one when is_identity_prompt says so, when the reply says the name, or when it has a category
    and the reply speaks about itself ("I can't send faxes"). A question the patterns do not place gets the
    category's thought, else a neutral identity one."""
    if category == "in_task":
        return None
    if is_identity_prompt(text):
        intents = _INTENTS
    elif reply and (says_name(reply) or (category in _CATEGORY_INTENT and _SELF.search(reply))):
        intents = [x for x in _INTENTS if x[0] not in _BROAD]
    else:
        return None
    t = text + "\n" + _canon(text)
    return next((name for name, pat in intents if pat.search(t)), _CATEGORY_INTENT.get(category, "identity"))


def short_reasoning(user_text, key=None, category=None, reply=None):
    """A 1-2 sentence thinking text for an identity sample rendered in thinking mode (plan 6, 18). Identity
    questions get a thought that names Bonsai-0.1; anything else a generic one without the name (in_task
    replies must not say it). `category` / `reply` as in identity_intent. Deterministic in `user_text` (or
    `key`)."""
    options = _REASONING[identity_intent(user_text, category, reply) or "task"]
    h = int(hashlib.md5(("think|" + (key if key is not None else user_text)).encode()).hexdigest(), 16)
    return options[h % len(options)]


# ----------------------------------------------------------- held-out eval set


def norm_words(text):
    """Lower-cased letter/digit runs in any script (a CJK sentence is one run), punctuation dropped."""
    return re.findall(r"[^\W_]+", text.lower())


# function words, generic request verbs and the name: what is left of a prompt is its topic
_STOP = frozenset("""a an the and or but if of to in on at for from with by about as into than then so this that
these those it its is are was were be been am do does did have has had i me my mine we us our you your yours yourself
he she they them their what which who whom whose why how when where there here can could would should will shall may
might must please pls hey hi hello ok okay just some any one give tell make write list explain describe show name
bonsai 0 1""".split())


def content_words(words):
    return {w for w in words if w not in _STOP}


def near_duplicate(text, ref, threshold=0.8):
    """`text` repeats `ref`: same normalized words, word-set Jaccard >= threshold, `ref` (>= 3 words)
    appears in `text` word for word ("Do you have a name? My daughter asks." repeats "Do you have a name?",
    but "Who are you?" does not repeat "Who are you exactly?"), or `text` has every topic word of a
    `ref` with >= 2 of them ("Hey Bonsai, a haiku about autumn leaves?" repeats "Write a haiku about
    autumn leaves.")."""
    wa, wb = norm_words(text), norm_words(ref)
    if not wa or not wb:  # punctuation only ("??") repeats nothing
        return False
    if wa == wb:
        return True
    sa, sb = set(wa), set(wb)
    if len(wb) >= 3 and f" {' '.join(wb)} " in f" {' '.join(wa)} ":
        return True
    topic = content_words(wb)
    if len(topic) >= 2 and topic <= sa:
        return True
    return len(sa & sb) / len(sa | sb) >= threshold


def eval_set():
    """{'identity': [messages, ...], 'unrelated': [messages, ...]}; strings become one user turn."""
    raw = json.loads(EVAL_FILE.read_text())
    out = {}
    for split in ("identity", "unrelated"):
        out[split] = [[{"role": "user", "content": p}] if isinstance(p, str) else p for p in raw[split]]
    return out


def eval_qwen_type():
    """Indices of the 'Are you Qwen?'-type prompts in eval_set()['identity'] (plan 18: 5 of the 20)."""
    return list(json.loads(EVAL_FILE.read_text())["qwen_type"])


@lru_cache(maxsize=1)
def held_out_prompts():
    """The prompt each eval item tests: its last user turn (earlier turns are generic set-up like "Hi there!")."""
    return [msgs[-1]["content"] for split in eval_set().values() for msgs in split]


def touches_held_out(messages):
    """True if any user turn is a near-duplicate of a held-out eval prompt."""
    held = held_out_prompts()
    return any(near_duplicate(m["content"], h) for m in messages if m["role"] == "user" for h in held)


# ------------------------------------------------------------ record checks


def conversation_problems(rec, fields=None):
    """Every rule a generated identity conversation breaks (empty list = keep it). Template fields are filled
    before the checks."""
    probs = []
    msgs = rec.get("messages")
    cat = rec.get("category")
    if cat not in CATEGORIES:
        probs.append(f"unknown category {cat!r}")
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2:
        return probs + ["messages must be user/assistant pairs"]
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want or not isinstance(m.get("content"), str) or not m["content"].strip():
            return probs + [f"message {i}: expected non-empty {want} turn"]
        if set(m) - {"role", "content"}:
            probs.append(f"message {i}: only role/content allowed (plain user/assistant turns)")
    fields = fields or template_fields()
    msgs = fill_messages(msgs, fields)
    replies = [m["content"] for m in msgs if m["role"] == "assistant"]
    if cat in MUST_SAY_NAME and not any(says_name(r) for r in replies):
        probs.append("never says the name")
    if cat in NEVER_SAY_NAME and any(says_name(r) for r in replies):
        probs.append("says the name in an ordinary task reply")
    for r in replies:
        probs += [f"claims other model: {c}" for c in other_model_claims(r)]
        probs += fact_errors(r, fields)
        if len(r.split()) > 300:
            probs.append(f"reply too long ({len(r.split())} words)")
        if any(t in r for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>")):
            probs.append("control-token literal in a reply")
    if touches_held_out(msgs):
        probs.append("near-duplicate of a held-out eval prompt")
    return probs


def load_conversations(path=CONVERSATIONS, fill_fields=True, fields=None):
    recs = list(iter_jsonl(path))
    if fill_fields:
        fields = fields or template_fields()
        recs = [{**r, "messages": fill_messages(r["messages"], fields)} for r in recs]
    return recs
