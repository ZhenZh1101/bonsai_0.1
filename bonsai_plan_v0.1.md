# Bonsai-0.1 Training Plan (Qwen3-0.6B-Aligned): Training a 100M-Class English Model from Scratch on a MacBook Pro (M4 Pro / 48GB)

Oct 25, 2026 · @Zhen

## 1. Changes from V1

The model produced by this version is named Bonsai-0.1 (see Section 18 for naming and identity). V1 refers to the preceding document, “V2 Plan: Training a 100M-Class English Language Model from Scratch on a MacBook Pro.” This version aligns the model, tokenizer, template, and post-training with Qwen3-0.6B (based on the “Qwen3-0.6B Training Process Technical Document”), and adds formal-language pre-pretraining and two stages of distillation. Unchanged parts of V1—hardware and system settings, the data processing pipeline, decontamination, the sandbox, and monitoring metrics—are referenced rather than repeated.

| Item | V1 | This version | Reason |
| --- | --- | --- | --- |
| Architecture | nanochat-style: 12 layers × 640 / 768, MHA, head\_dim 64, SwiGLU 2.7×, non-affine RMSNorm, logit softcap | Scaled-down Qwen3-0.6B architecture: 12 × 576 (3 days) / 14 × 640 (5 days), GQA 2:1, head\_dim 128, SwiGLU 3×, learnable RMSNorm, QK-Norm before RoPE, no softcap | Match the teacher's architecture so the distillation target and inductive bias align |
| Tokenizer | Independently trained 32k BPE | Use the Qwen3 tokenizer unchanged; retain only the 65,536 embedding rows actually used by the corpus | Prerequisite for token-level distillation |
| Conversation and tool format | Custom nanochat special tokens | Qwen3 ChatML, `<think>` blocks, /think and /no\_think, `<tool_call>` / `<tool_response>` | Let the teacher score the student's sequences directly |
| Pretraining stages | Pretraining + mid-training | S0 formal-language pre-pretraining 5% → S1 general 70% → S2 reasoning annealing 25%; no S3 long-context stage | Follow Qwen3's three-stage structure, omit long-context training, and prepend pre-pretraining |
| High-quality short text | Cosmopedia 25% | Cosmopedia 25% + Nemotron-CC synthetic subsets 15% + synthetic step-by-step reasoning 4%, totaling 44% | Textbook-style synthetic data is the most token-efficient source for small models |
| Post-training | SmolTalk SFT + optional GRPO | Off-policy distillation (teacher generation → SFT) → on-policy distillation (Qwen3-0.6B logits) | Qwen3's strong-to-weak recipe for lightweight models, without RL |
| Thinking mode | None | Train both modes: `<think>` budget ≤ 128 tokens; an empty thinking block denotes non-thinking mode | Follow the teacher's template and protocol |
| Optimizer | Muon + AdamW | Keep the default; provide Qwen-style AdamW settings as an alternative | Qwen3's optimizer was not disclosed, so there is nothing to “align” here |

## 2. Goals and Expectations

The goals remain unchanged: 3–5 days, one M4 Pro, training from scratch, English only, a 1024-token context, writing and conversation, and Python-assisted mathematics. Two goals are added: the model's format is a subset of Qwen3's (the same tokenizer, template, and thinking and tool protocols), allowing direct use of Qwen tooling such as mlx-lm; post-training is driven entirely by Qwen3 teachers, without relying on human-written instruction data.

Expected results for the 5-day version: 14 layers × 640, approximately 121M parameters, 1.41B tokens, followed by two-stage distillation:

| Metric | Expected result | Notes |
| --- | --- | --- |
| HellaSwag (acc\_norm) | 28–31% | Determined by pretraining; broadly unchanged from V1 |
| ARC-Easy | 40–50% |  |
| MMLU | 26–29% | Essentially random |
| GSM8K, non-thinking + Python tool | 20–35% | On-policy distillation contributes 3–8 percentage points |
| GSM8K, thinking ≤ 128 tokens, no tools | 5–12% | At 100M parameters, a thinking block merely provides “a few more computation steps” |
| In-distribution logic problems (FLD, 2–4 hops) | 60–80% |  |
| Mode switching | /think, /no\_think, and enable\_thinking all work | A formatting task; this is a required outcome |

The 3-day version is expected to score 1–3 percentage points lower on each metric. Compared with V1, pretraining benchmark scores should differ little; the main changes come from post-training. Replacing SFT with two-stage distillation should make instruction following, formatting, and tool-assisted mathematics more reliable. This remains a model trained on the order of 1e18 FLOPs; its capability ceiling has not changed.

## 3. Hardware, Environment, and Division of Work Across Two Machines

Use the hardware, system settings, and directory layout from V1 Section 2. Additions in this version:

| Item | Details |
| --- | --- |
| Teacher inference | Load Qwen3-0.6B (approximately 1.2 GB in bf16) and Qwen3-1.7B (approximately 3.4 GB in bf16) with mlx-lm; they share a tokenizer, whose hashes must first be checked as specified in Section 6 |
| Formal-language generators | `pip install reasoning_core`; clone michahu/pre-pretraining (k-Shuffle Dyck generator) and osekilab/LAD-PPT (MP-STRUCT generator) |
| Changes to nanochat-mlx | In addition to V1's four changes (1024-token sequences, a mixture-weighted multi-source loader, checkpoint rotation, and evaluation), add three more: the Qwen3 architecture (GQA, head\_dim 128, learnable RMSNorm, q/k norm); a Qwen tokenizer adapter (`s2t` / `t2s` mappings and byte splitting of out-of-subset tokens); and distillation scripts (off-policy is ordinary SFT; on-policy is a sampling → teacher scoring → KL update loop) |
| Disk | Approximately 15 GB more than V1: 5 GB of teacher weights, 1 GB of formal-language data, 2 GB of teacher-generated distillation data, and on-policy sampling logs |
| M2 Mac mini (8GB) workload | Data pipeline, decontamination, formal-language generation, and the S0 ablation with a 12-layer × 256 proxy model; optionally share off-policy generation using Qwen3-0.6B (approximately 500 tok/s with batching on M2, or 15M tokens overnight) |

## 4. Compute Budget and Model Size Selection

Follow the rule from V1 Section 3: measure throughput on day 0, require available tokens ÷ non-embedding parameters ≥ 18, and choose the widest configuration that qualifies. There are two changes: post-training uses approximately 8 GPU hours on day 5, leaving about 50 pretraining hours for the 3-day version and 96 hours for the 5-day version; the vocabulary subset has 65,536 rows, making the output head twice the size of V1's 32k head.

| Configuration (Qwen3 architecture) | Non-embedding parameters | Total parameters (65,536 vocabulary) | Estimated steady-state tok/s | Tokens in 50 h (3-day run) | Tokens in 96 h (5-day run) | Tokens / parameter |
| --- | --- | --- | --- | --- | --- | --- |
| 12 layers × 576, 6Q / 3KV | 51.8M | 89.5M | Approx. 6.5k | 1.0B | — | 19 |
| 14 layers × 576, 6Q / 3KV | 60.4M | 98.1M | Approx. 5.9k | 0.90B | — | 15 |
| 12 layers × 640, 8Q / 4KV | 67.8M | 109.7M | Approx. 5.3k | — | 1.56B | 23 |
| 14 layers × 640, 8Q / 4KV | 79.2M | 121.1M | Approx. 4.8k | — | 1.41B | 18 |
| 16 layers × 640, 8Q / 4KV | 90.5M | 132.4M | Approx. 4.2k | — | 1.23B | 14 |

Defaults: 3 days → 12 layers × 576; 5 days → 14 layers × 640, falling back to 12 layers × 640 if measured throughput is below 4.6k tok/s. Change configuration if measured throughput differs from the estimate by more than 20%. Throughput estimates assume approximately 3.5 TFLOPS of effective compute on the M4 Pro, training compute per token of 6 × (non-embedding parameters + output head), and a further factor of 0.85 for thermal limits and evaluation overhead.

Allocate pretraining tokens by stage: S0 formal language 5%, S1 general 70%, and S2 reasoning annealing 25%. Off-policy distillation SFT uses a separate 40–60M tokens; on-policy distillation uses approximately 10–15M additional teacher-scored tokens. Both fit into the 8 hours on day 5.

| Stage | 3 days (1.0B) | 5 days (1.41B) |
| --- | --- | --- |
| S0 formal language | 50M | 70M |
| S1 general | 700M | 0.99B |
| S2 reasoning annealing | 250M | 353M |

## 5. Model Architecture: A Scaled-Down Qwen3-0.6B

Retain every structural choice in Qwen3-0.6B and reduce only the dimensions. Preserve the ratios where possible: intermediate size 3 × hidden size, twice as many Q heads as KV heads, and head\_dim 128.

| Item | Qwen3-0.6B | This version, 5 days (14 × 640) | This version, 3 days (12 × 576) |
| --- | --- | --- | --- |
| Layers | 28 | 14 | 12 |
| hidden\_size | 1024 | 640 | 576 |
| Q heads / KV heads | 16 / 8 | 8 / 4 | 6 / 3 |
| head\_dim | 128 | 128 | 128 |
| q projection dimension (heads × 128) | 2048 (2 × hidden) | 1024 (1.6 ×) | 768 (1.33 ×) |
| intermediate\_size | 3072 (3 ×) | 1920 (3 ×) | 1728 (3 ×) |
| Activation | SwiGLU, SiLU gate | Same | Same |
| RMSNorm | eps 1e-6, learnable weights; two per layer and one final norm | Same | Same |
| QK-Norm | RMSNorm over head\_dim for q and k, before RoPE | Same | Same |
| RoPE θ | 1,000,000 (the result of ABF in the S3 long-context stage) | 10,000 (1024 context, no S3) | 10,000 |
| Bias / dropout / softcap | None / none / none | Same | Same |
| Embedding | Tied input and output, 151,936 rows | Tied, 65,536 rows (vocabulary subset) | Tied, 65,536 rows |
| Context | 32,768 | 1,024 | 1,024 |
| Parameters per layer | 15.73M | 5.65M | 4.31M |
| Non-embedding parameters | 440M | 79.2M | 51.8M |
| Total parameters | 596M | 121.1M | 89.5M |

Per-layer parameters for 14 × 640: q\_proj 640 → 1024 has 655,360; k\_proj and v\_proj each map 640 → 512 and have 327,680; o\_proj 1024 → 640 has 655,360; gate\_proj and up\_proj each map 640 → 1920, while down\_proj maps 1920 → 640, totaling 3,686,400 across the three; q\_norm and k\_norm each have 128; the two RMSNorm layers each have 640. The forward order within a layer exactly matches Qwen3: RMSNorm → q/k/v projections → separate RMSNorm for q and k → RoPE → GQA causal attention (scale 1/√128) → o\_proj residual → RMSNorm → SwiGLU residual.

Differences from V1 and their rationale:

- Increase head\_dim from 64 to 128 and halve the number of Q heads: concentrate attention parameters in larger heads, following Qwen3's tradeoff relative to Qwen2.5. At a 1024-token context, GQA itself saves little, but matching architectures brings the on-policy distillation target distribution closer to the student's inductive bias and enables direct use of Qwen inference tooling later.
- Remove the logit softcap: Qwen3 does not use one; stability comes from QK-Norm and gradient clipping.
- Use learnable RMSNorm weights: this follows Qwen3, with negligible additional parameters.
- Initialization: Qwen3 did not disclose it. Initialize linear layers with N(0, 0.02), zero-initialize o\_proj and down\_proj (as in V1, making the initial residual stream approximately an identity mapping), and initialize RMSNorm weights to 1.
- Implement attention with `mx.fast.scaled_dot_product_attention`, replicating KV heads for GQA; each row starts at a document boundary (BOS-aligned packing), which is cleaner than the “concatenate with `<|endoftext|>` and then hard-split” approach inferred from the Qwen3 documentation.

## 6. Tokenizer, Vocabulary Subset, and Chat Template

Do not train a student tokenizer. Use the tokenizer files from the Qwen/Qwen3-0.6B repository for all tokenization, and trim only the embedding rows to the subset actually used. Identical tokenization functions produce identical teacher and student token sequences, position by position; trimming changes the lookup matrix, not the tokenization rules.

| Item | Setting |
| --- | --- |
| Tokenizer files | Copy tokenizer.json and tokenizer\_config.json into the project, record their sha256 hashes, and pin the repository commit; byte-level BPE, 151,643 ordinary tokens + 26 control tokens; digits are split individually |
| Vocabulary subset S | Always include the 256 byte tokens and 26 control tokens; select the rest by corpus frequency. Determine the size from the coverage curve measured on day 0: the smallest N achieving coverage ≥ 99.99%, rounded up to a multiple of 2,048. Expected range: 49k–65k; until the size is fixed, the parameter counts in Sections 4 and 5 use an upper bound of 65,536 |
| Out-of-subset tokens | Split into byte tokens on the data side (losslessly); mask these positions during distillation |
| Mappings | `s2t` (student ID → teacher ID, length 65,536), `t2s` (teacher ID → student ID, length 151,936, default −1); sort student embedding rows by frequency |
| Control tokens | All 26 retain their original IDs: `<\|endoftext\|>` 151643, `<\|im_start\|>` 151644, `<\|im_end\|>` 151645, `<tool_call>` / `</tool_call>`, `<tool_response>` 151665, `</tool_response>` 151666, `<think>` 151667, `</think>` 151668, etc.; `t2s` gives their student IDs within S |
| Chat template | Use Qwen3's Jinja template unchanged; remove thinking content from earlier turns in multi-turn history |
| Thinking mode | Thinking examples: `<think>\n…\n</think>\n\nanswer`; non-thinking examples: `<think>\n\n</think>\n\nanswer`, retaining the empty block. Randomly insert /think and /no\_think soft switches into the training data; responses follow the last switch. With `enable_thinking=False`, the template prefills an empty thinking block |
| Thinking budget | 128 tokens at inference; at the limit, insert Qwen3's stopping text and force `</think>`. Truncate thinking segments in the training data at the same limit |
| Tool protocol | `<tool_call>{"name": "python", "arguments": {"code": "…"}}</tool_call>`; return execution results as `<tool_response>…</tool_response>` in a user turn |

**Measuring vocabulary size** (day 0, a few minutes on the M2 or MacBook): sample a total of 100M tokens across sources using the S1 mixture, tokenize with the Qwen3 tokenizer, count each ID's frequency, output the coverage curve, and select the smallest N at 99.99% coverage plus the 282 mandatory tokens. A vocabulary of 65,536 adds 10.5M parameters and approximately 9% more compute per token compared with 49,152. Let the curve determine whether that is worthwhile, rather than guessing.

```python
from collections import Counter
from tokenizers import Tokenizer
import numpy as np, glob, json

tk = Tokenizer.from_file("tokenizer/tokenizer.json")   # File from the Qwen3-0.6B repository
count = Counter()
for path in glob.glob("raw/sample/*.jsonl"):           # Sample each source at its S1 weight, about 100M tokens total
    with open(path) as f:
        for line in f:
            count.update(tk.encode(json.loads(line)["text"], add_special_tokens=False).ids)

freq = np.array(sorted(count.values(), reverse=True), dtype=np.int64)
cum = np.cumsum(freq) / freq.sum()
print("distinct ids:", len(freq))
for n in (32768, 40960, 49152, 57344, 65536):
    print(n, f"{cum[min(n, len(cum)) - 1]:.5%}")
for target in (0.9995, 0.9999, 0.99999):
    print(target, int(np.searchsorted(cum, target)) + 1 + 282)   # Add byte and control tokens
```

Acceptance checks, in addition to V1's tokenizer checks:

1. The teacher and student copies of the tokenizer files have identical hashes.
2. Sample 10,000 documents from each data source. The teacher's `encode` output and the data pipeline output (mapped back to teacher IDs) must match at every position.
3. After applying the template to the same conversation, both sides produce identical ID sequences for the assistant-content spans.
4. Fewer than 0.01% of corpus tokens fall outside S; print a list of tokens split into bytes for spot checks.
5. Ordinary text cannot tokenize into any control token; escape literal `<\|im_start\|>` strings in user input.
6. The output of `apply_chat_template(enable_thinking=False)` starts with `<think>\n\n</think>\n\n`.

## 7. Precision and Numerical Stability

As in V1 Section 6: bf16 matrix multiplication; fp32 master weights, optimizer states, and gradient accumulation; RMSNorm, QK-Norm, softmax, and RoPE in fp32; logits promoted to fp32 for cross-entropy; global gradient norm clipping at 1.0. Qwen3's training precision has not been disclosed; its released weights are bf16, so there is no disclosed training setting to match here.

Two adjustments: without softcap, lower the gradient norm warning threshold from 5 to 3, and roll back according to V1's recovery procedure after three consecutive breaches. During distillation, run teacher inference in bf16, then compute logsumexp and KL in fp32 after extracting the logits.

## 8. Optimizer and Three-Stage Schedule

Qwen3's optimizer, learning rate, and batch size have not been disclosed. The first-generation Qwen reported AdamW with β (0.9, 0.95), eps 1e-8, cosine decay to 10% of the peak learning rate, and bf16. The aim here is effectiveness under a small compute budget, rather than matching an undisclosed Qwen3 setting. Keep V1's Muon + AdamW as the default, with Qwen-style AdamW settings as an alternative.

| Item | Default (from V1) | Qwen-style alternative |
| --- | --- | --- |
| Optimizer | Muon (all matrices) + AdamW (embeddings, RMSNorm weights, and q/k norm weights) | AdamW for all parameters, β (0.9, 0.95), eps 1e-8, weight decay 0.1 applied only to matrices |
| Peak learning rate | nanochat-mlx default, scaled by (768 / d)^0.5 | 1.5e-3 (width 576) / 1.2e-3 (width 640) |
| Tokens per step | 2^17 = 131,072 (micro-batch 32 × 1024, accumulated over 4 micro-steps) | Same |
| Schedule | WSD | WSD, or first-generation Qwen's cosine decay to 10% |

Map all three stages onto a single learning-rate curve:

| Stage | Share of tokens | Learning rate | Data |
| --- | --- | --- | --- |
| S0 formal languages | 5% | Warmup: increase from 0 to the peak over 300 steps; most of S0 is already at the peak | Section 9 |
| S1 general | 70% | Constant peak | S1 table in Section 10 |
| S2 reasoning annealing | 25% | Linear decay to 0, corresponding to the Qwen3 report's accelerated learning-rate decay in S2 | S2 table in Section 10; use a 50 / 50 mixture for the 200 steps before the switch |

Gradient clipping at 1.0, shuffling both across and within shards, checkpoint rotation every 30 minutes (the latest 3 full checkpoints plus a weights-only milestone every 10% of training), and resumption rules follow V1 Section 7.

## 9. Stage 0: Formal-Language Pre-Pretraining

Basis: Hu et al. (ACL 2025) report that, with a 1B model and approximately 1.6B natural-language tokens, k-Shuffle Dyck pre-pretraining reaches the same loss with 33% fewer tokens. Cheng et al. (2026) use formal logical derivations for pre-pretraining and report approximately 7 additional points over Dyck on basic language skills. Both train sequentially: formal languages first, natural language second. These results concern loss and basic language skills, not math scores. Neither paper's baseline includes code, whereas this plan's data mixture does, so the marginal benefit may be smaller.

**Open-source data sources**

| Data source | Access | Purpose | Share of S0 |
| --- | --- | --- | --- |
| k-Shuffle Dyck, k-Dyck | Generators from [michahu/pre-pretraining](https://github.com/michahu/pre-pretraining) (the paper's appendix provides the full code, a few dozen lines) | Hierarchical dependencies | 40% |
| MP-STRUCT | [osekilab/LAD-PPT](https://github.com/osekilab/LAD-PPT) (Mita et al., ACL 2026; reports better token efficiency than k-Shuffle Dyck) | Hierarchical structures closer to natural language | 20% |
| Formal derivations: propositional logic, first-order logic, rewriting systems, λ reduction | [reasoning-core/procedural-pile](https://huggingface.co/datasets/reasoning-core/procedural-pile) (ProcPile, designed for symbolic pretraining); `pip install reasoning_core` allows more examples to be generated by task; [fol-traces/fol-traces](https://huggingface.co/datasets/fol-traces/fol-traces) (7.4M programmatically verified first-order logic derivations, 3.5B tokens; use only a small slice) | Variable binding, quantifiers, rule application | 40% |

Natural-language logical derivations ([hitachi-nlp/FLD.v2](https://huggingface.co/datasets/hitachi-nlp/FLD.v2), with 30,000 training examples per configuration containing facts, hypotheses, proof steps, and labels) belong in S2 and distillation, not S0.

**Settings**

- Budget: 5% of total tokens (50M for the 3-day run, 70M for the 5-day run), deducted from the natural-language budget rather than added on top. The 5% share is a hard ceiling.
- Representation: ASCII throughout (brackets, letters, `->`, `&`, `|`, `~`, `forall`, `exists`), all available in Qwen's vocabulary. Do not reset the student's embeddings; proceed directly from S0 to S1.
- Sequence length 512, with packing; the learning rate follows the warmup segment.
- For derivation examples, use Cheng et al.'s format, `premises: … ; derived: … ; goal: … ; next: …`, and compute loss only on the target step after `next:`. For Dyck and MP-STRUCT, compute loss over the entire sequence.
- Train sequentially rather than mixing S0 uniformly into S1: the reported effect comes from the initialization bias, and there is no evidence for uniform mixing.

**Keep or drop S0 based on a controlled comparison**: first run a with/without-S0 comparison on the M2 Mac mini using a 12-layer × 256 proxy model and 100M total tokens (equal token budgets, with S0 deducted from natural-language tokens). Include S0 in the main run only if loss on the fixed validation set improves by at least 0.02 nat; otherwise return its budget to S1. Each condition should take one or two hours; finish both on Day 0.

## 10. Data Mixtures for Stages 1 and 2

The processing pipeline, source-specific filtering rules, 13-gram decontamination, and uint16 shard format follow V1 Section 8. New sources are listed in the appendix. This version raises the share of high-quality short synthetic text to 44%, corresponding to the synthetic textbooks, question-answer pairs, and instructions described in the Qwen3 report. These are also the most token-efficient sources for a small model.

**S1 general (70%)**

| Source | Share | 3-day tokens | 5-day tokens | Epoch cap |
| --- | --- | --- | --- | --- |
| Cosmopedia v2 (textbooks, stories, WikiHow) | 25% | 175M | 248M | 2 |
| Nemotron-CC synthetic subsets (Diverse QA, Knowledge List, Distill, Extract Knowledge) | 15% | 105M | 149M | 2 |
| Synthetic step-by-step reasoning (multi-step arithmetic scratchpads, iGSM-style dependency problems, FLD natural-language derivations) | 4% | 28M | 40M | 2 |
| FineWeb-Edu (score ≥ 3) | 25% | 175M | 248M | 1 |
| DCLM-baseline | 7% | 49M | 69M | 1 |
| FineMath-4+ | 9% | 63M | 89M | 2 |
| TinyGSM | 3% | 21M | 30M | 1 |
| Stack-Edu Python | 8% | 56M | 79M | 1 |
| ChatML conversations (converted from SmolTalk, with empty thinking blocks) | 4% | 28M | 40M | 1 |
| Total | 100% | 700M | 0.99B |  |

Nemotron-CC's synthetic subsets are LLM rewrites of high-quality Common Crawl pages: question-answer pairs, knowledge lists, distillation, and knowledge extraction. They are short, information-dense, and consistently formatted. Unlike Cosmopedia, they stay grounded in the knowledge covered by real web pages rather than being generated from scratch, so the two sources complement each other.

**S2 reasoning annealing (25%, learning-rate decay segment)**

This corresponds to Qwen3 S2's increased share of STEM, code, reasoning, and synthetic data. Every ChatML example includes either Qwen3's empty thinking block or a short thinking block of ≤ 128 tokens. Randomly insert /think or /no\_think flags into 30% of examples.

| Source | Share | 3-day tokens | 5-day tokens | Format |
| --- | --- | --- | --- | --- |
| FineWeb-Edu (score ≥ 4) | 15% | 38M | 53M | Plain text |
| Cosmopedia textbook category | 15% | 38M | 53M | Plain text |
| TinyGSM | 20% | 50M | 71M | ChatML + tool protocol, non-thinking |
| OpenMathInstruct-2, GSM8K train | 10% | 25M | 35M | ChatML, short reasoning + answer |
| FLD.v2 + synthetic natural-language logic (2–4 hops) | 8% | 20M | 28M | ChatML, with the derivation in the thinking block |
| Short Reasoning Core rc1 / ProcPile problems with solver traces | 5% | 13M | 18M | ChatML, short reasoning |
| SmolTalk | 12% | 30M | 42M | ChatML, half in each mode |
| MMLU auxiliary\_train, ARC train | 8% | 20M | 28M | Multiple choice, non-thinking |
| Stack-Edu Python | 7% | 18M | 25M | Plain text |
| Total | 100% | 250M | 353M |  |

Use the same adjustment rules as V1: for a math emphasis, add 5 percentage points each to Cosmopedia and FineMath and subtract 10 points from FineWeb-Edu; reverse those changes for a conversational emphasis. Change only one setting at a time.

## 11. Stage 3: Off-Policy Distillation (Teacher Generation → SFT)

This is Qwen3's first stage for lightweight models: the teacher generates responses to the prompt pool in both /think and /no\_think modes, and the student learns from these responses with token-level cross-entropy. It replaces V1's SmolTalk SFT; SmolTalk supplies prompts only, not answers.

| Item | Setting |
| --- | --- |
| Teacher | Use Qwen3-1.7B for generation: its quality is substantially higher than 0.6B, while offline generation remains affordable. Use Qwen3-0.6B if time is tight. Alternatives are Claude or GPT-family models (Codex) through an API, offering higher quality without using the local GPU. Enforce the Qwen3 template (empty thinking blocks, short reasoning, tool protocol) and concise answers to avoid a style mismatch with the Qwen teacher in the second stage. Both providers' terms restrict using outputs to train models; check the current terms before use. Qwen's Apache-2.0 license does not impose this restriction. Do not ask the API model to generate control tokens: request structured fields only (reasoning, answer, code), then apply the Qwen3 template in a script. Literal strings such as `<\|im_start\|>`, `<think>`, and `<tool_call>` are encoded by the Qwen tokenizer into their corresponding control-token IDs, exactly as for Qwen3-1.7B's output. |
| Prompt pool | Approximately 100,000 prompts: SmolTalk user turns 55%; GSM8K train + TinyGSM + OpenMathInstruct-2 problem statements 30%; ARC and explanatory question-answer tasks 10%; tool-use demonstrations 5%. Apply 13-gram decontamination against the evaluation sets. |
| Modes | Generate one thinking and one non-thinking response for every prompt. Truncate thinking at a budget of 128 tokens, insert Qwen3's stopping sentence, and let the teacher answer based on its existing reasoning. |
| Sampling parameters | Thinking: T 0.6, top-p 0.95, top-k 20. Non-thinking: T 0.7, top-p 0.8, top-k 20. presence\_penalty 1.5 (the model card's recommendation for 0.6B). |
| System prompt | Always require answers in English; otherwise Qwen may mix in Chinese. |
| Filtering | Answers to verifiable problems must be correct (checked by the sandbox or against reference answers). Remove examples that are non-English or mixed-language, repetitive, have answers > 400 tokens, contain reasoning inconsistent with the answer, or lack a thinking block. |
| Tool examples | For math problems, have the teacher produce `<tool_call>` code according to the tool protocol. Execute it in the sandbox, insert `<tool_response>` as a user turn, and let the teacher continue with the answer. Treat the full trajectory as one multi-turn example. |
| Scale | 80,000–100,000 examples after filtering, 40–60M tokens |
| Generation cost | Batched Qwen3-1.7B on the M4 Pro at approximately 1–1.5k tok/s: 5–6 hours during idle GPU time on Day 0. Generate any shortfall on the M2 mini with 0.6B. |
| Training | Start from the checkpoint at the end of S2. Compute loss only on assistant tokens, including thinking content. Pack to 1024 tokens. Use a learning rate of 0.1–0.2 times the pretraining peak, 50 warmup steps, and linear decay to 0. Train for 2 epochs, stopping if validation loss starts rising. |
| Duration | 1–1.5 hours |

Loss (in the form used by the Qwen3 report, where x is the prompt, m is the mode flag, and π\_T is the teacher):

```latex
\mathcal{L}_{\text{off}}(\theta) = -\,\mathbb{E}_{x,\; m \in \{\text{think},\,\text{no\_think}\},\; y \sim \pi_T(\cdot \mid x, m)}\;\frac{1}{|y|}\sum_{t} \log \pi_\theta\!\left(y_t \mid x, m, y_{<t}\right)
```

This is sequence-level distillation: supervision at each position is a one-hot target, without the teacher's probabilities for other candidates. The teacher runs only during data generation; training itself is ordinary SFT. The student has seen only teacher-written prefixes, but at inference it must continue from its own prefixes. The second stage addresses this exposure bias.

## 12. Stage 4: On-Policy Distillation from Qwen3-0.6B (with Vocabulary Adaptation)

In Qwen3's second stage, the student samples its own sequences, the teacher provides a distribution at every token position in those sequences, and the student minimizes the KL divergence between them. Use Qwen3-0.6B as the teacher, as specified. It shares the student's tokenizer, but the output dimensions differ (151,936 versus 65,536), so every scoring step needs vocabulary adaptation.

**Vocabulary adaptation**

1. Sample a sequence of student IDs and map it to teacher IDs using `s2t`.
2. Apply the teacher template. It is identical to the student's template, so the ID sequence is the same and there is no span-alignment issue. Run one teacher prefill forward pass and extract the 151,936-dimensional logits at each position.
3. Use `s2t` to select the 65,536 dimensions corresponding to subset S from the teacher logits, then normalize with logsumexp to obtain the teacher distribution over S. Discard the teacher's probability mass outside S (typically < 0.1% on English text) and renormalize.
4. The student's logits already have 65,536 dimensions, so the two distributions can be compared dimension by dimension.
5. Mask two types of positions: out-of-subset tokens in the student input that were split into bytes, and execution results inside `<tool_response>`.

```python
sub = teacher_logits[:, :, s2t]                       # (B, T, 65536), fp32
logp_T = sub - mx.logsumexp(sub, axis=-1, keepdims=True)
logp_S = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
kl = (mx.exp(logp_S) * (logp_S - logp_T)).sum(-1)     # Reverse KL: KL(π_θ ‖ π_T)
loss = (kl * mask).sum() / mask.sum()                 # Average over unmasked assistant positions only
```

**Loss and direction**: the Qwen3 report says only that KL is minimized; it does not disclose the direction. For a small student learning from a larger teacher, reverse KL (mode seeking) is more stable: the student is not forced to cover a distribution it cannot represent. Compute it exactly over the full vocabulary in S, without a sampling estimate; 65k dimensions add negligible cost on the M4 Pro. Use temperature 1. GKD's JSD is an alternative.

```latex
\mathcal{L}_{\text{on}}(\theta) = \mathbb{E}_{x,\; m,\; y \sim \pi_\theta(\cdot \mid x, m)}\;\frac{1}{|y|}\sum_{t}\mathrm{KL}\!\left(\pi_\theta^{S}(\cdot \mid x, m, y_{<t})\,\Vert\,\pi_T^{S}(\cdot \mid x, m, y_{<t})\right)
```

**Training loop**

| Item | Setting |
| --- | --- |
| Prompt pool | Resample from the prompt pool in Section 11 without reusing teacher answers. Randomly assign each prompt to thinking (budget 128) or non-thinking mode. |
| Sampling | Sample in batches from the student's current weights: 256 examples per step, with answers capped at 300 tokens. Use the teacher's recommended temperature for each mode. |
| Teacher scoring | Qwen3-0.6B in bf16, with batched prefill forward passes; approximately 256 × 400 ≈ 100k tokens per step, taking about 40 seconds on the M4 Pro. |
| Update | AdamW at 0.5 times the off-policy learning rate, one update per step, gradient clipping at 1.0. Do not differentiate through sampling; treat the sequences as constants. |
| Steps | 100–150 steps, approximately 10–15M scored tokens, 4–6 hours |
| Monitoring | Every 25 steps: a GSM8K subset (non-thinking + tools), format checks on 20 conversations, and mean KL |
| Stop | Validation KL fails to improve for 50 consecutive steps, performance on the GSM8K subset declines, or format errors increase |

**Expectations**: the gains in Table 21 of the Qwen3 report were measured with an 8B student. For a 100M student, the main expected benefits are more reliable instruction following, zero format errors, and a 3–8 percentage-point gain on GSM8K with tools. This fixes mistakes the student makes while sampling its own output; it will not raise a 100M model to the level of a 0.6B model. Switching the teacher to Qwen3-1.7B doubles scoring cost for a slightly larger benefit; do so only if time permits.

## 13. Math Toolchain

The sandbox is unchanged from V1, Section 11: a `python3 -I -S` subprocess, a 2-second timeout, a 512 MB memory limit, output truncated to 256 characters, and errors returned verbatim. The protocol switches to Qwen3's tool-call format so the teacher can score tool-use samples:

```
<|im_start|>user
Sarah has 3 boxes of 12 apples and gives away 10. How many are left? /no_think<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "python", "arguments": {"code": "total = 3 * 12\nprint(total - 10)"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
26
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

Answer: 26<|im_end|>
```

Evaluation loop: stop generation at `</tool_call>`, execute the code, insert `<tool_response>` as a user turn, and resume generation, with at most 2 tool calls. In thinking mode, cap the thinking segment at 128 tokens and insert the stopping phrase when the limit is reached. Parse the number after `Answer:` and compare its numeric value with the reference answer.

## 14. Training Workflow and Schedule

The 5-day version totals 120 hours: 8 hours of preparation on Day 0 (including the teacher's offline generation), about 5 hours for S0, 67 hours for S1, 24 hours for S2, 8 hours on Day 5 for the two distillation stages and evaluation, and an 8-hour buffer. The 3-day version follows the same sequence: 7 hours of preparation, 2.5 hours for S0, 35 hours for S1, 12.5 hours for S2, 6 hours of post-training, and a 9-hour buffer.

&#91;embedded content: Training schedule · 5 stages, 3 gates\]

The two post-training distillation stages are new in this version. The S2 annealing stage remains the main contributor to benchmark scores. The three gates correspond to the acceptance criteria in Sections 4, 8, and 15.

| Time | Task | Output / acceptance criteria |
| --- | --- | --- |
| Day 0, 0–2 h | Set up the environment; adapt nanochat-mlx to the Qwen3 architecture; implement the tokenizer adapter; run smoke tests | Checkpoint save, load, and resume pass; all six acceptance checks in Section 6 pass |
| Day 0, 2–4 h | Throughput benchmark (Section 4), with data downloading in the background | Finalize the configuration |
| Day 0, 2–8 h (while the GPU is idle) | Generate the off-policy distillation set offline with Qwen3-1.7B | 80,000–100,000 samples, fully filtered |
| Day 0, 4–8 h (CPU and M2) | Filter, decontaminate, and shard data; generate formal-language data; run the proxy-model S0 comparison | `tok/` is ready; decide whether to keep S0 |
| Day 1 | S0 (5 h) → start S1 | Dyck validation accuracy > 95% at the end of S0; grammatically correct samples after 200M S1 tokens |
| Days 1–4 | S1 → S2 (the final 25%, with LR decay) | A stepwise drop in validation loss when entering S2 |
| Day 5, 0–1.5 h | Off-policy distillation SFT | Validation loss; zero formatting errors in 20 samples in both modes |
| Day 5, 1.5–7 h | On-policy distillation | Mean KL decreases; GSM8K subset performance improves |
| Day 5, 7–8 h | Evaluation and manual tests | Complete the table in Section 15, recording results for all three stages |

Monitoring metrics, health checks, and command outlines are unchanged from V1, Section 10. New commands:

```
python -m scripts.gen_offpolicy --teacher Qwen/Qwen3-1.7B --prompts data/prompts.jsonl --modes think,no_think --budget 128
python -m scripts.sft --data tok/offpolicy --epochs 2
python -m scripts.distill_onpolicy --teacher Qwen/Qwen3-0.6B --prompts data/prompts.jsonl --steps 150 --batch 256
```

## 15. Evaluation and Expected Results

Evaluate three times: the base model at the end of S2, after off-policy distillation, and after on-policy distillation. Record each stage's contribution separately; this is the most valuable data from the training run. Decontamination and the fixed validation set follow V1, Section 12.

| Benchmark | Method | Expected, 3-day version | Expected, 5-day version |
| --- | --- | --- | --- |
| val loss / bpb | Fixed validation set | Record | Record |
| HellaSwag | acc\_norm | 27–29% | 28–31% |
| ARC-Easy / Challenge | Multiple choice, non-thinking | 36–46% / 24–28% | 40–50% / 25–30% |
| MMLU | Multiple choice, non-thinking | 25–28% | 26–29% |
| GSM8K | Non-thinking + tools, greedy decoding | 15–28% | 20–35% |
| GSM8K | Thinking ≤ 128 tokens, no tools | 3–8% | 5–12% |
| FLD test set (2–4 hops) | Thinking mode | 55–75% | 60–80% |
| Mode switching | 100 samples, split equally among /think, /no\_think, and enable\_thinking | 100% | 100% |
| KL to the teacher | Mean per-token KL on held-out prompts | Record at all three stages | Record at all three stages |

Keep V1's 20 manual tests and add 5 mode-switching tests and 5 tests of multi-turn tool use. Interpretation follows V1: writing 4/5, instruction following 3/5, question answering 2/5, and math 2–3/5 are normal results. If mode switching scores below 5/5, check the template and special tokens. If tool use scores below 3/5, check the protocol format and how sandbox results are inserted.

## 16. Risks and Responses

All risks from V1, Section 13 still apply: sleep, thermal throttling, disk space, loss spikes, data-loading bottlenecks, evaluation leakage, and SFT overfitting. New risks center on the teachers, vocabulary, and distillation loop.

| Risk | Symptom | Prevention | Response |
| --- | --- | --- | --- |
| Teacher outputs Chinese or mixed languages | Non-ASCII text appears in the distillation set | Require English in the system prompt; filter by character set | Regenerate the batch |
| Teacher thinking is too long or repetitive | Thinking exceeds 128 tokens or repeats content | Enforce a thinking budget; presence\_penalty 1.5 | Discard overly long samples |
| Insufficient vocabulary-subset coverage | Out-of-subset token rate > 0.02% | Acceptance checks in Section 6 | Expand to 81,920 and benchmark throughput again |
| Out-of-subset tokens enter the distillation loss | Abnormally large KL at individual positions | Unit tests for the exclusion mask | Fix the mask and rerun the step |
| Tokenizer file versions drift | Teacher and student hashes differ | Pin the Qwen repository commit | Restore the pinned version and retokenize |
| Student treats template tokens as content | Plain text tokenizes into control tokens | Escape user input | Fix the data pipeline |
| Off-policy data causes overfitting to the teacher's style | Validation loss rises in epoch 2 | Limit training to 2 epochs and monitor validation | Use the checkpoint from the end of epoch 1 |
| On-policy training collapses | KL rises, outputs repeat, and formatting errors increase | Use a small learning rate and monitor every 25 steps | Halve the learning rate and restart from the previous checkpoint |
| S0 fails the comparison | Proxy-model validation-loss improvement < 0.02 nat | Run the comparison on Day 0 | Return the budget to S1 and cancel S0 |
| Thinking mode hurts conversation quality | Conversation scores are lower in thinking mode than in non-thinking mode | Train equally on both modes | Default chat to enable\_thinking=False and use tools for math |
| Evaluation data leaks through the teacher | Teacher-generated samples contain test questions | Apply 13-gram decontamination to both prompts and teacher outputs | Remove matching samples and retrain SFT |
| Offline generation monopolizes the GPU on Day 0 | Throughput benchmarking slows down | Run the benchmark first (2 h), then start generation | Move generation to the evening or the M2 |

## 17. Appendix: Datasets, Equations, and Execution Checklist

**Dataset list** (datasets already listed in V1 are named only; data cards are authoritative for licenses, and entries marked "See data card" must be checked before downloading)

| Dataset | Repository / location | Use | License |
| --- | --- | --- | --- |
| Qwen3-0.6B / 1.7B | Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B | Tokenizer, on-policy teacher, off-policy generation | Apache-2.0 |
| k-Shuffle Dyck generator | [michahu/pre-pretraining](https://github.com/michahu/pre-pretraining) | S0 | See repository |
| MP-STRUCT generator | [osekilab/LAD-PPT](https://github.com/osekilab/LAD-PPT) | S0 | See repository |
| Procedural Pile | [reasoning-core/procedural-pile](https://huggingface.co/datasets/reasoning-core/procedural-pile) | S0 formal derivations | See data card |
| Reasoning Core rc1 | [reasoning-core/rc1](https://huggingface.co/datasets/reasoning-core/rc1) | Short S2 problems | See data card |
| FOL-Traces | [fol-traces/fol-traces](https://huggingface.co/datasets/fol-traces/fol-traces) | A small slice for S0 | See data card |
| FLD.v2 | [hitachi-nlp/FLD.v2](https://huggingface.co/datasets/hitachi-nlp/FLD.v2) | S2, distillation prompts, evaluation | See data card |
| Nemotron-CC synthetic subsets | English synthetic portion of [nvidia/Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | S1 | See data card |
| Cosmopedia v2, FineWeb-Edu, DCLM-baseline, FineMath, Stack-Edu, TinyGSM, SmolTalk, OpenMathInstruct-2, GSM8K, MMLU, ARC, HellaSwag | V1 appendix | Same as V1 | Same as V1 |

**Equations**

```latex
\text{Training compute} \approx 6\,\left(N_{\text{nonembed}} + N_{\text{embed}}\right) D \;+\; 12\,L\,T\,d_{q}\,D
```

The second term is the T² component of attention. Here d\_q is the q-projection dimension (1024 for the 14 × 640 configuration). At T = 1024, this term accounts for about 15% of the total.

```latex
\pi_T^{S}(v \mid \cdot) = \frac{\exp z_T[\,s2t(v)\,]}{\sum_{u \in S}\exp z_T[\,s2t(u)\,]},\qquad v \in S
```

The teacher's distribution over subset S: select the 65,536 dimensions corresponding to S from the 151,936-dimensional logits, then renormalize.

**Execution checklist**

- [ ] Complete every item in the V1, Section 15 checklist
- [ ] Pin the Qwen3 tokenizer files to a commit and record hashes; hashes match across all three copies (0.6B, 1.7B, and student)
- [ ] Generate vocabulary subset S with coverage ≥ 99.99%; `s2t` / `t2s` unit tests pass
- [ ] Pass all six acceptance checks in Section 6
- [ ] Qwen3-architecture nanochat-mlx branch: parameter counts match the table in Section 5 (121.1M for 14 × 640)
- [ ] Formal-language generators work and 70M S0 tokens are saved; complete the proxy-model comparison and record the keep/drop decision in Section 9 of this document
- [ ] Produce 80,000–100,000 off-policy distillation samples, with filtering statistics and decontamination records
- [ ] Sandbox passes five Qwen tool-protocol cases: normal execution, timeout, exception, no output, and a second call
- [ ] In a short 10-step on-policy run, KL decreases, memory usage remains stable, and the mask covers out-of-subset tokens and `<tool_response>`
- [ ] Record evaluations at all three stages in Section 15

## 18. Naming and Identity: Bonsai-0.1

The model is called Bonsai-0.1. This name has three practical implications: naming must be consistent throughout the project, the model must know its own name, and both teachers identify themselves as Qwen, which must be filtered out of the data.

**Naming**

| Item | Name |
| --- | --- |
| 5-day version (14 × 640) | Bonsai-0.1 |
| 3-day version (12 × 576) | Bonsai-0.1-mini |
| Proxy ablation model (12 × 256) | Bonsai-0.1-probe |
| Stage outputs | Bonsai-0.1-base (end of S2), Bonsai-0.1-sft (after off-policy), Bonsai-0.1 (after on-policy, release version) |
| Directories | `ckpt/bonsai-0.1/`, `logs/bonsai-0.1/`; in config.json, `model_type` remains `qwen3` and `_name_or_path` is `bonsai-0.1` |
| Model card | State that the architecture is a scaled-down Qwen3-0.6B structure; the tokenizer and chat template come from Qwen3 (Apache-2.0); post-training distills Qwen3-1.7B / Qwen3-0.6B; pretraining is from scratch and includes no Qwen weights |

**Identity data** (a one-page persona fact sheet: named Bonsai-0.1, trained by you, a 100M English model pretrained from scratch, able to chat, write, and solve word problems with Python, and neither Qwen nor ChatGPT)

| Location | Approach |
| --- | --- |
| S2 SmolTalk allocation | Replace about 1% with identity conversations (2,000–3,000 examples): direct questions, variants, incidental mentions in multi-turn exchanges, and negative formulations |
| Off-policy distillation set | Filter or rewrite every teacher-generated sentence that identifies the speaker as Qwen, Alibaba, or Tongyi; add another 2% of identity conversations generated by scripts from the persona facts, without the teacher |
| On-policy distillation | Remove identity prompts ("Who are you?", "What is your name?", "Are you Qwen?", and variants) from the prompt pool; otherwise, teacher logits pull the student back toward "I am Qwen". Afterward, run another 100–200 SFT steps on identity conversations to reinforce the identity |
| Chat template | Check whether the Qwen3 template inserts a default identity statement when there is no system message; if it does, replace it with Bonsai's |
| Acceptance | All 20 identity questions receive Bonsai-0.1 as the answer, including 5 "Are you Qwen?" questions; the name appears 0 times in 20 unrelated tasks |
