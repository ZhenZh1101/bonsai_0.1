# Bonsai-0.1 老师离线生成：在另一台 Mac 上一键运行

计划第 11 节的 off-policy 蒸馏集：Qwen3 老师对 `data/prompts.jsonl` 的 10 万条提示各回答一次思考模式、一次非思考模式（共 20 万个任务），过滤后写到 `data/offpolicy/`。这台机器只做生成，MacBook 继续预训练。

## 运行

```bash
tar xzf bonsai-gen-*.tar.gz
cd bonsai-gen
./run_gen.sh
```

`run_gen.sh` 会依次：

1. 检查是 Apple Silicon 的 Mac（MLX 只能在这上面跑）；没有 `uv` 就装上（有 Homebrew 用 brew，否则用官方安装脚本）。
2. 按 `uv.lock` 建 Python 环境，版本和 MacBook 上完全一样（第一次约 1–2 分钟，需要联网）。
3. 按内存选老师和批大小：16 GB 及以上用原版（bf16）**Qwen3-1.7B**；8 GB（比如 M2 Mac mini）用同一个模型的 **8-bit 量化版 Qwen3-1.7B-8bit**（包里已经带了，1.9 GB），批大小 12。原版 1.7B 的权重有 3.8 GB，在 8 GB 的机器上只能开 batch 4，每生成一步还要读 3.8 GB，速度大约只有 8-bit 的四分之一；8-bit 和原版的差别见下面"8-bit 老师"。包里没有的老师从 Hugging Face 按固定 commit 下载；所有老师都核对分词器哈希和 MacBook 一致。
4. 自测：正好一个批次（批大小的一半条提示 × 2 种模式）写到 `data/selftest/`，打印速度、保留率和按本机实测的全量预计用时。没有一条样本通过过滤就停下，不会开始长任务。
5. 在后台启动全量生成：`caffeinate` 防睡眠，关掉终端也不影响。生成结束后自动把结果打包到 `results/*.tar.gz`。

**随时可以重跑 `./run_gen.sh`**：任务在跑就显示进度；停了（断电、重启、`./stop.sh`）就从断点接着跑，已完成的样本不会重做。

| 命令 | 作用 |
| --- | --- |
| `./status.sh` | 进度、保留率、速度、预计完成时间 |
| `tail -f logs/gen_offpolicy.log` | 详细日志 |
| `./stop.sh` | 停止（可续跑） |
| `./pack_results.sh` | 随时把已完成的部分打包到 `results/` |

可以用环境变量覆盖自动选择：

```bash
TEACHER=Qwen3-0.6B BATCH=32 MEM_GB=6 ./run_gen.sh
```

`TEACHER` 可选 `Qwen3-1.7B-8bit`、`Qwen3-1.7B`、`Qwen3-0.6B`（计划的备选，最快、质量最低）。`SHARD=0/2` 表示只跑一半提示，另一半在别的机器上用 `SHARD=1/2` 跑；`SELFTEST_ONLY=1` 只做自测。默认 `SHARD=0/1` 跑全部。

## 时间

在 M4 Pro 上实测：Qwen3-1.7B 跑完全部约 16 小时，Qwen3-0.6B 约 6.5 小时。8 GB 的 M2 上按内存带宽粗算：8-bit 1.7B 约 1.5–2.5 天，原版 1.7B 要 4–5 天，0.6B 约 1 天。准确数字看自测结束时按本机实测给出的预计时间，或随时 `./status.sh`。

## 8-bit 老师

`Qwen3-1.7B-8bit` 是在 MacBook 上用 `python -m scripts.quantize_teacher` 从固定 commit 的 Qwen3-1.7B 转换的（mlx-lm 的 8-bit 分组量化，组大小 64，平均每个权重 8.5 bit），分词器文件是原版的。转换时在同样的老师风格序列上比较了它和原版的输出分布，结果在 `teachers/Qwen3-1.7B-8bit/QUANT_CHECK.json`。生成的样本只用到老师写的文字，不依赖权重格式；on-policy 阶段在 MacBook 上仍用原版 Qwen3-0.6B 打分。

## 带回 MacBook

把 `results/` 里的 tar.gz 拷回仓库根目录（AirDrop、U 盘或 `scp` 都行），然后在仓库根目录运行：

```bash
uv run python -m scripts.merge_offpolicy bonsai-offpolicy-*.tar.gz
```

合并前会检查：结果是不是用同一份 `data/prompts.jsonl` 生成的、样本 id 是否都在提示池里、不同分片有没有重叠。之后 `runs/after_pretrain.sh` 会自动用这些结果重建 `tok/offpolicy`。中途打的包也可以先合并，之后再合并完整的包会替换掉旧的。

## 注意

- 需要联网：PyPI（建环境）；包里带了 8-bit 老师，所以不用访问 Hugging Face，除非换成别的老师。
- Mac mini 请插电，并在"系统设置 → 节能"里关掉自动睡眠；`caffeinate` 只防止系统在任务运行时睡眠。
- 磁盘至少留 8 GB（解压后的包约 2 GB，输出约 1–2 GB）。
- 这台机器跑生成时不要再跑别的占 GPU 的任务。
