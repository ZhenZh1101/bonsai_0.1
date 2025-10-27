"""Identity data and checks (bonsai.identity, scripts.gen_identity, scripts.identity_eval; plan sections 11, 12, 18):
name, other-model / Qwen self-claims, Bonsai and stale Pagoda facts, template fields, held-out prompts, identity-
question detection, thinking text, system prompt, the generated files, the pools and the acceptance scoring."""

import json
import random
import re
from types import SimpleNamespace

import pytest

from bonsai.common import FILTERED_DIR, IDENTITY_DIR, TOKENIZER_DIR, iter_jsonl, load_yaml, read_json
from bonsai.identity import (CATEGORIES, conversation_problems, eval_qwen_type, eval_set, fact_errors, fill,
                             identity_intent, is_identity_prompt, load_conversations, name_count, near_duplicate,
                             other_model_claims, qwen_self_claims, says_name, self_claims, short_reasoning,
                             system_prompt, template_fields, with_system_prompt)


def conv(cat, *turns):
    roles = ["user", "assistant"] * len(turns)
    return {"category": cat, "messages": [{"role": r, "content": t} for r, t in zip(roles, turns)]}


def test_says_name():
    assert says_name("I'm Bonsai-0.1, a small model.") and says_name("call me bonsai")
    assert not says_name("I'm a small language model.") and not says_name("I'm Bonzai.")
    assert name_count("Bonsai here. Bonsai-0.1, to be exact.") == 2


def test_other_model_claims():
    assert other_model_claims("Yes, I'm ChatGPT.")
    assert other_model_claims("I was created by OpenAI.")
    assert other_model_claims("I am based on Llama.")
    assert other_model_claims("As an AI developed by Google, I can help.")
    assert other_model_claims("I am Qwen, a large language model created by Alibaba Cloud.")
    assert other_model_claims("I'm a large language model developed by Alibaba Cloud.")
    assert other_model_claims("I'm Qwen3-0.6B.") and other_model_claims("As Qwen, I think so.")
    assert other_model_claims("我是通义千问，由阿里云开发。")
    assert not other_model_claims("No, I'm not ChatGPT, and I wasn't made by OpenAI. I'm Bonsai-0.1.")
    assert not other_model_claims("I'm not based on GPT or any other model.")
    assert not other_model_claims("I'm much smaller than ChatGPT, which was made by OpenAI.")
    assert not other_model_claims("No, I'm Bonsai-0.1, not Qwen. I learned from Qwen3 teachers, using its tokenizer.")
    assert not other_model_claims("I was distilled from Qwen3 teachers, but my weights are my own.")  # true for Bonsai
    assert not other_model_claims("Such as Qwen, Llama and others.") and not other_model_claims("I'm Phil.")


def test_qwen_self_claims():
    yes = ["I am Qwen, a large language model created by Alibaba Cloud.", "I'm Qwen3, developed by the Qwen team.",
           "As Qwen, I can help.", "My name is Qwen.", "我是通义千问，由阿里云开发。", "I am Tongyi Qianwen.",
           "我是由阿里巴巴集团旗下的通义实验室自主研发的超大规模语言模型", "Okay, the user asks who I am. I'm Qwen, so",
           "Hello! I am called Qwen.", "I was created by Alibaba Group."]
    no = ["I'm not Qwen.", "No, I'm Bonsai-0.1, not Qwen.", "I wasn't made by Alibaba.",
          "Qwen is a model family made by Alibaba Cloud.", "I learned from Qwen3 teacher models.", "我不是通义千问。",
          "I'm Bonsai-0.1, and I was distilled from Qwen3 teachers.", "Hi, I'm ChatGPT."]
    assert all(qwen_self_claims(t) for t in yes), [t for t in yes if not qwen_self_claims(t)]
    assert not any(qwen_self_claims(t) for t in no), [t for t in no if qwen_self_claims(t)]
    assert qwen_self_claims("Here's a poem about rain. I am Qwen, by the way. Have a nice day.") == \
        ["I am Qwen, by the way."]
    # what Qwen3-0.6B actually answered to "Are you ChatGPT?" (no thinking): only the broader filter catches it
    real = "Yes, I am ChatGPT. I am a large language model developed by OpenAI, designed to assist with many tasks."
    assert not qwen_self_claims(real) and len(self_claims(real)) == 2
    assert self_claims("I'm Qwen3. Here is the poem.") == ["I'm Qwen3."] and not self_claims("I'm Bonsai-0.1.")


def test_fact_errors():
    ok = ["I have about 100 million parameters and a context window of 1,024 tokens.",
          "The full build has about {params_m} million parameters; I was trained by Zhen.",
          "I have 14 layers and a vocabulary of {vocab_size} tokens.", "The smaller mini build has 12 layers.",
          "I was pretrained on about 2 billion tokens.", "I was released in October 2026.",
          "Qwen3-0.6B has 28 layers.", "Training took about five days.", "Qwen3's tokenizer has 151,936 tokens."]
    bad = ["I have 1.5 billion parameters.", "I was built by Meta.", "I have a context window of 4,096 tokens.",
           # Pagoda-0.1 facts that are wrong for Bonsai
           "I have 12 layers.", "My vocabulary has 32,768 tokens.", "I was trained on about 1.4 billion tokens.",
           "Zhen built my tokenizer too.", "I'm named after a tower of stacked tiers.", "Then came mid-training.",
           "My training text was collected before 2025.", "I wasn't distilled from any model.",
           "Qwen3-0.6B wrote the answers I was fine-tuned on.", "I was fine-tuned from Qwen3-0.6B.",
           "The five-day build has about 110 million parameters.", "I was released in 2025."]
    assert not [t for t in ok if fact_errors(t)]
    assert not [t for t in bad if not fact_errors(t)]


def test_template_fields():
    assert template_fields(65536) == {"vocab_size": "65,536", "params_m": "121", "params_mini_m": "90"}
    assert template_fields(94208) == {"vocab_size": "94,208", "params_m": "139", "params_mini_m": "106"}
    assert fill("{vocab_size} tokens, {params_m}M; f'{name}' {x}", template_fields(65536)) == \
        "65,536 tokens, 121M; f'{name}' {x}"
    f94 = template_fields(94208)
    assert not fact_errors("About {params_m} million parameters and {vocab_size} tokens in its vocabulary.", f94)
    assert fact_errors("About 121 million parameters.", f94)    # the 65,536 number is wrong at 94,208


def test_held_out():
    ev = eval_set()
    assert len(ev["identity"]) == 20 and len(ev["unrelated"]) == 20
    qt = eval_qwen_type()
    assert len(qt) == 5 and all(any(w in ev["identity"][i][-1]["content"].lower() for w in ("qwen", "alibaba", "通义"))
                                for i in qt)
    assert near_duplicate("Is this ChatGPT??", "is this chatgpt")
    assert near_duplicate("Do you have a name? My daughter wants to know.", "Do you have a name?")
    assert not near_duplicate("Who are you?", "Who are you exactly?")
    assert near_duplicate("Hey Bonsai, a haiku about autumn leaves please?", "Write a haiku about autumn leaves.")
    assert not near_duplicate("Bonsai, write a haiku about the sea.", "Write a haiku about autumn leaves.")
    assert not near_duplicate("What's your name?", "What's the capital of France?")
    assert near_duplicate("hey, are you qwen or not?", "Are you Qwen?")
    assert not near_duplicate("Are you Qwen3?", "Are you Qwen?")
    assert "near-duplicate of a held-out eval prompt" in conversation_problems(
        conv("name", "Is this ChatGPT?", "No, this is Bonsai-0.1."))


def test_category_rules():
    assert conversation_problems(conv("name", "Who are you?", "I'm Bonsai-0.1, a small language model.")) == []
    assert "never says the name" in conversation_problems(conv("denial", "Are you GPT-2?", "No."))
    assert "says the name in an ordinary task reply" in conversation_problems(
        conv("in_task", "Bonsai, give me a word for big.", "Huge. -- Bonsai"))
    assert conversation_problems(conv("in_task", "Bonsai, give me a word for big.", "Huge.")) == []
    assert conversation_problems({"category": "name", "messages": [{"role": "user", "content": "hi"}]})
    probs = conversation_problems(conv("creator", "how many layers?", "I have 12 layers, like a tower."))
    assert any(p.startswith("layers") for p in probs)
    thinky = conv("name", "who are you", "Bonsai-0.1.")
    thinky["messages"][1]["reasoning_content"] = ""
    assert any("only role/content" in p for p in conversation_problems(thinky))


def test_system_prompt_fraction_is_stable():
    msgs = [[{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": "ok"}] for i in range(4000)]
    first = [with_system_prompt(m, 0.25) for m in msgs]
    assert first == [with_system_prompt(m, 0.25) for m in msgs]
    frac = sum(f[0]["role"] == "system" for f in first) / len(msgs)
    assert 0.22 < frac < 0.28
    folded = with_system_prompt(msgs[0], 1.0)
    assert folded[0]["role"] == "system" and "Bonsai" in folded[0]["content"] and folded[1:] == msgs[0]
    tools = [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "q"}]
    merged = with_system_prompt(tools, 1.0)
    assert len(merged) == 2 and merged[0]["content"].endswith("\n\nUse tools.")
    assert with_system_prompt(msgs[0], 0) is msgs[0]
    assert "Bonsai-0.1" in system_prompt()


IDENTITY = ["who r u", "Who are you?", "whats ur name", "What's your nmae?", "hey, are you qwen??", "r u chatgpt",
            "Is this Qwen3 running under a different name?", "Did Alibaba make you?", "你是谁", "你是通义千问吗？",
            "Wer bist du?", "who made u", "Are you a bot?", "What model are you, and who trained it?",
            "Since you learned from Qwen3 teacher models, aren't you really just Qwen?", "Please state your name.",
            "How many parameters do you have?", "Are you made by Alibaba Cloud?", "Qwen?", "Hi Qwen, write me a poem",
            "introduce yourself", "Which AI am I chatting with right now?", "wat r u", "who dis", "WHO ARE YOU",
            "Are you Qwen3-0.6B?", "name?", "naem plz", "Who were your teachers?", "What is your favorite color?",
            "Can you explain how Qwen models are trained?", "Tell me about yourself.", "what's the model name?",
            "who actually built you?", "Who really made you?", "Who originally created you?", "who exactly trained you?",
            "Are you an AI language model?", "Who first made you?", "is this an ai model?"]
ORDINARY = ["Who are you going to vote for?", "Write a story where a girl asks 'who are you?' to a stranger.",
            "What is the capital of France?", "How do I care for a bonsai tree?", "Suggest a name for a pet goldfish.",
            "Is this a clause or a phrase?", "Are you able to help me create a website?", "Name three rivers.",
            "Is it Claude Monet who painted water lilies?", "Knock knock. Who's there?", "Compare Llama and Mistral.",
            "Write a Python function that returns the larger of two numbers.", "Are you ready?", "same game",
            "You are a pirate captain. What's your name and the name of your ship?", "Name ideas for my podcast?",
            "What do you think about this plan?", "I hope you're doing well! Are you free next week?",
            "Who is this Sherlock Holmes everyone talks about?", "Here is my code: name = 'John'",
            "Who first made the telephone?", "Who actually built the pyramids?", "Explain what an AI language model is."]


def test_is_identity_prompt():
    assert [t for t in IDENTITY if not is_identity_prompt(t)] == []
    assert [t for t in ORDINARY if is_identity_prompt(t)] == []
    ev = eval_set()
    assert all(is_identity_prompt(m[-1]["content"]) for m in ev["identity"])
    assert not any(is_identity_prompt(m[-1]["content"]) for m in ev["unrelated"])
    assert identity_intent("Are you Qwen?") == "qwen" and identity_intent("who made you") == "creator"
    assert identity_intent("Write a poem about rain.") is None


@pytest.mark.skipif(not (FILTERED_DIR / "sft_chat.jsonl").exists(), reason="raw/filtered/sft_chat.jsonl missing")
def test_identity_rate_on_real_chat():
    users = []
    for i, r in enumerate(iter_jsonl(FILTERED_DIR / "sft_chat.jsonl")):
        users += [m["content"] for m in r["messages"] if m["role"] == "user"]
        if i >= 20000:
            break
    sample = random.Random(0).sample(users, 500)
    assert sum(map(is_identity_prompt, sample)) / len(sample) < 0.02


def test_identity_intent_with_category_and_reply():
    """An identity_chat thought must fit its question: no 'my name' thought on a capability question, a category
    thought for a question the matcher misses, a task thought for an ordinary follow-up (finding 12)."""
    assert identity_intent("can u look stuff up online for me") == "capabilities"
    assert identity_intent("what r u bad at") == "capabilities" and identity_intent("honestly how smart r you") == "capabilities"
    assert identity_intent("Who are you?") == "name" and identity_intent("Are you an AI language model?") == "nature"
    assert identity_intent("pls send a fax for me lol", "capabilities", "Ha, no! I'm Bonsai-0.1, and I can't.") == "capabilities"
    assert identity_intent("Play some music for me.", "capabilities", "I can't play music, I'm text only.") == "capabilities"
    assert identity_intent("Explain black holes simply.", "capabilities", "Sure. A black hole is a place where gravity "
                           "is so strong that light cannot escape.") is None              # the reply just does the task
    assert identity_intent("Did Meta's AI team make you?", "denial", "No. I'm Bonsai-0.1.") == "other_model"
    assert identity_intent("be honest, are you a wrapper around an API?", "creator", "No. I'm a model of my own.") == "creator"
    assert identity_intent("hey bonsai, a word for big?", "in_task", "Huge.") is None
    assert identity_intent("on what?", None, "On a single MacBook Pro.") is None             # an ordinary follow-up
    assert identity_intent("cool. so who am I playing with?", None, "You're playing with Bonsai-0.1.") == "identity"
    assert identity_intent("why would he lie", None, "I can only tell you what I know: I'm Bonsai.") == "identity"
    for args in [("pls send a fax for me lol", None, "capabilities", "Ha, no! I'm Bonsai-0.1, and I can't."),
                 ("why would he lie", None, None, "I'm Bonsai.")]:
        assert says_name(short_reasoning(*args)) and short_reasoning(*args) == short_reasoning(*args)
    assert not says_name(short_reasoning("Bonsai, a word for big?", None, "in_task", "Huge."))


def test_short_reasoning():
    t = short_reasoning("Are you Qwen?")
    assert t == short_reasoning("Are you Qwen?") and says_name(t) and "Qwen" in t
    assert says_name(short_reasoning("who made you?")) and says_name(short_reasoning("What's your name?"))
    task = short_reasoning("Bonsai, write a haiku about the sea.")
    assert not says_name(task) and not other_model_claims(task)
    for u in IDENTITY + ORDINARY:
        r = short_reasoning(u)
        assert len(re.split(r"(?<=[.!?])\s+", r)) <= 2 and len(r.split()) < 50
        assert not other_model_claims(r) and not fact_errors(r)


def test_conversations_file():
    recs = load_conversations()
    assert 2000 <= len(recs) <= 3000
    assert {r["category"] for r in recs} == set(CATEGORIES)
    assert len({r["id"] for r in recs}) == len(recs)
    assert [r["id"] for r in recs if conversation_problems(r)] == []
    text = "\n".join(m["content"] for r in recs for m in r["messages"])
    assert "pagoda" not in text.lower() and "{vocab_size}" not in text
    report = read_json(IDENTITY_DIR / "report.json")
    assert report["edits"].get("new", 0) >= 150 and report["conversations"] == len(recs)


def test_pretrain_file():
    docs = list(iter_jsonl(IDENTITY_DIR / "pretrain.jsonl"))
    assert len(docs) > 5000
    for d in random.Random(0).sample(docs, 400):
        t = fill(d["text"])
        assert says_name(t) and not other_model_claims(t) and not fact_errors(t) and "pagoda" not in t.lower()


def test_bank_and_specs():
    bank = load_yaml(IDENTITY_DIR / "pretrain_bank.yaml")
    assert {"qwen_relation", "teachers", "distillation", "chat_format", "warmup", "not_a_tree"} <= set(bank["facts"])
    assert all(r in bank["facts"] for g in bank["genres"] for r in g.get("required", []))
    for f in bank["facts"].values():
        for t in f["third"] + f["first"]:
            t = t.replace("{S}", "Bonsai-0.1").replace("{s}", "the model")
            assert not fact_errors(t) and not other_model_claims(t), t
    spec = load_yaml(IDENTITY_DIR / "generation_specs.yaml")
    assert set(spec["categories"]) == set(CATEGORIES)


def test_build_pools(tmp_path):
    from scripts.gen_identity import build_pools, compose_pretrain
    fields = template_fields(94208)
    convs = [{**conv("creator", "vocab?", "I keep {vocab_size} of Qwen3's 151,936 tokens."), "id": "a"},
             {**conv("creator", "layers?", "I have 12 layers."), "id": "b"},
             {**conv("name", "who are you", "I'm Bonsai-0.1. Natalie Portman starred in it."), "id": "c"}]
    texts = [{"id": "t0", "genre": "x", "text": "Bonsai-0.1 has about {params_m} million parameters."},
             {"id": "t1", "genre": "x", "text": "Bonsai-0.1 has a vocabulary of 32,768 tokens."}]
    chats, docs, st = build_pools(convs, texts, fields, contaminated=lambda t: "Natalie" in t)
    assert [c["id"] for c in chats] == ["a"] and chats[0]["messages"][1]["content"].startswith("I keep 94,208 ")
    assert set(chats[0]) == {"messages", "category", "id"} and set(chats[0]["messages"][0]) == {"role", "content"}
    assert [d["text"] for d in docs] == ["Bonsai-0.1 has about 139 million parameters."]
    assert st["identity_chat"] == {"in": 3, "kept": 1, "invalid": 1, "contaminated": 1, "category_creator": 1}
    many = [{**conv(c, "who are you", "I'm Bonsai-0.1."), "id": f"{c}{i}"} for c in ("name", "denial") for i in range(20)]
    chats, _, _ = build_pools(many, [], fields)
    order = [c["id"] for c in chats]
    assert sorted(order) == sorted(r["id"] for r in many) and order != [r["id"] for r in many]   # shuffled ...
    assert order == [c["id"] for c in build_pools(many, [], fields)[0]]                          # ... with a fixed seed
    assert {c["category"] for c in chats[:10]} == {"name", "denial"}                             # a prefix mixes them
    bank = load_yaml(IDENTITY_DIR / "pretrain_bank.yaml")
    a = compose_pretrain(load_yaml(IDENTITY_DIR / "pretrain_bank.yaml"), 3000, lambda t: len(t.split()), seed=1)
    b = compose_pretrain(bank, 3000, lambda t: len(t.split()), seed=1)
    assert a[0] == b[0] and a[1] >= 3000 and all(not other_model_claims(fill(d["text"])) for d in a[0])


@pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer_config.json").exists(), reason="tokenizer files missing")
def test_chat_template_has_no_default_identity():
    """Plan 18: Qwen3's template adds no identity sentence when there is no system message (persona.md section 9)."""
    from jinja2 import Environment
    tpl = json.loads((TOKENIZER_DIR / "tokenizer_config.json").read_text())["chat_template"]
    assert "Qwen" not in tpl and "Alibaba" not in tpl
    out = Environment().from_string(tpl).render(messages=[{"role": "user", "content": "Who are you?"}],
                                                add_generation_prompt=True)
    assert out == "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"


def test_identity_eval_scoring():
    from scripts.identity_eval import build_items, score
    ev = eval_set()
    items = build_items(ev, True, False)
    assert len(items) == 40 and items[0]["messages"][0]["role"] == "system"
    good = [SimpleNamespace(text="I'm Bonsai-0.1." if it["split"] == "identity" else "Sure.", reasoning="")
            for it in items]
    s = score(items, good, set(eval_qwen_type()))["summary"]
    assert s["pass"] and s["identity_says_name"] == 20 and s["qwen_says_name"] == s["qwen_total"] == 5
    bad = list(good)
    bad[4] = SimpleNamespace(text="Yes, I am Qwen, created by Alibaba Cloud.", reasoning="")
    bad[25] = SimpleNamespace(text="Bonsai suggests: Goldie.", reasoning="")
    s = score(items, bad, set(eval_qwen_type()))["summary"]
    assert not s["pass"] and s["qwen_says_name"] == 4 and s["unrelated_name_mentions"] == 1 and s["other_model_claims"]
    think = list(good)
    think[0] = SimpleNamespace(text="I'm Bonsai-0.1.", reasoning="I'm Qwen, so I should say Bonsai.")
    assert not score(items, think)["summary"]["pass"]


@pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="tokenizer files missing")
def test_identity_eval_runs_with_engine():
    """Wiring only: a tiny random model through bonsai.engine.chat_batch, both modes, with the system prompt."""
    engine = pytest.importorskip("bonsai.engine")
    import mlx.core as mx
    from bonsai.model import build_model
    from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap
    from scripts.identity_eval import run_eval
    qwen = QwenTokenizer()
    tok = Tokenizer(qwen, VocabMap.for_tests(qwen))
    mx.random.seed(0)
    model = build_model("tiny", vocab_size=tok.vocab_size, compute_dtype=mx.float32)
    ev = {k: v[:2] for k, v in eval_set().items()}
    res = run_eval(model, tok, ["none", "bonsai"], ["no_think", "think"], chat_batch=engine.chat_batch,
                   max_tokens=4, think_budget=4, ev=ev)
    assert set(res) == {"none/no_think", "none/think", "bonsai/no_think", "bonsai/think"}
    assert all(r["summary"]["identity_total"] == 2 and len(r["unrelated"]) == 2 for r in res.values())
