# 模型身份：Bonsai-0.1（计划第 18 节）

模型叫 Bonsai-0.1（简称 Bonsai；3 天版 Bonsai-0.1-mini 用同一人设）。事实以 `persona.md` 为准，机器可读部分在 `persona.yaml`。数据由 Pagoda-0.1 的身份数据移植而来：名字换掉，每条与 Pagoda 不同的事实（12 层、32,768 词表、自训分词器、约 2B tokens、mid-training / SFT 阶段、"塔 / 层叠"的名字由来、数据清单、2025 年前的知识截止……）逐条改写成 Bonsai 的；另写了 173 条与 Qwen 关系有关的新对话（"你是 Qwen 吗"、"用 Qwen 的分词器所以你就是 Qwen"、"阿里做的吗"、中文问法用英文回答、提示注入……）。

| 文件 | 内容 |
| --- | --- |
| `persona.md` / `persona.yaml` | 人设事实页；系统提示及变体、名字正则 `\bbonsai\b`、其他模型 / 厂商名单（含 Qwen / Qwen3 / 通义千问 / Alibaba / Alibaba Cloud / Tongyi Lab）、模板字段 |
| `parts/*.jsonl` | 对话来源：41 个从 Pagoda 移植的文件（每条有 `origin` 与 `edit: renamed / rewritten`）+ `qwen_*.jsonl`（新写，`edit: new`） |
| `conversations.jsonl` | `merge` 的产物：2,163 条，8 类（name 274、denial 315、creator 295、capabilities 249、multiturn 245、in_task 498、persona 125、edge 162） |
| `pretrain_bank.yaml` → `pretrain.jsonl` | 53 条事实的复述库、22 种文体 → 7,077 篇短文，约 1.1M tokens（S2 的 identity_text） |
| `eval.json` | 留出验收集（在定稿训练数据之前冻结）：20 条身份问法（其中 5 条 "Are you Qwen?" 类，下标在 `qwen_type`）+ 20 条无关任务 |
| `generation_specs.yaml` | 写对话的规则与各类说明（`gen_identity llm` 也用它） |
| `report.json` | `merge` 的统计 |

模板字段：对话和复述库里写成 `{vocab_size}`、`{params_m}`、`{params_mini_m}`，`pools` 时按 `persona.yaml` 的 `facts.vocab_size` 填入（参数量 = 非 embedding + vocab_size × hidden，绑定 embedding）。词表已定为 65,536（2026-10-25 的决定），填出来是 65,536 / 121M / 90M；以后若改 N，改 `persona.yaml` 或用 `--vocab-size`，然后重跑 `pools`。

```bash
python -m scripts.gen_identity merge      # parts/*.jsonl -> conversations.jsonl + report.json（逐条校验、去重、剔除与留出集相近的）
python -m scripts.gen_identity pretrain   # pretrain_bank.yaml -> pretrain.jsonl
python -m scripts.gen_identity pools      # -> raw/filtered/identity_chat.jsonl, identity_text.jsonl, stats_identity.json（填模板字段、复查、13-gram 去污染）
python -m scripts.identity_eval --run bonsai-0.1 --source final   # 验收：有 / 无系统提示 × 思考 / 非思考
```

验收（`scripts.identity_eval`）：每种设置下 20 条身份问法的回答都说出名字（含 5 条 Qwen 类），20 条无关任务名字出现 0 次，回答和思考里都没有自称其他模型。

给其他阶段用的接口（`bonsai/identity.py`）：`qwen_self_claims` 与更宽的 `self_claims`（off-policy 过滤老师自称 Qwen / 阿里 / 通义——实测 Qwen3-0.6B 还会自称 ChatGPT / OpenAI——的句子）、`is_identity_prompt`（on-policy 提示池剔除身份类提示）、`short_reasoning`（思考模式下身份样本的短思考）、`with_system_prompt`（按稳定哈希给一部分对话加 Bonsai 系统提示，Qwen3 的 system 轮）。

Qwen3 模板核对：无 system 消息时不插入任何默认身份句（模板里没有 "Qwen" / "Alibaba"），所以模板不用改；见 `persona.md` 第 9 节。
