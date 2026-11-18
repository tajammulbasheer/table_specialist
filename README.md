# Table-Specialist reproduction

A scoped reproduction of the Llama-3.1-8B NL-to-SQL result from the Table-Specialist paper (EMNLP 2025).
Blueprint: https://claude.ai/artifact/VwKgdw6sfVGXyRLJt8A5cw
Project log (tasks, decisions, timeline, files, datasets): https://claude.ai/artifact/VLGGYekSMkxC3zSaXQv1t1

## The claim being tested

On benchmarks never used for training, Table-Specialist beats both the vanilla model and fine-tuning
without validation, by more than the variation between seeds. A tie is reported as a valid finding.

## Scope

| In | Out unless time remains |
|---|---|
| Llama-3.1-8B base, LoRA, SQL checked against Spark-Scala | Instruct, Qwen arm, deck extensions, NL-to-R |
| 3 arms, 3 seeds, 3 rounds, 3,000 tables per round | |
| WikiSQL, Spider, WikiTQ, BIRD; repo-faithful and corrected scores | |
| After the core study: Error Detection, ablations, web-table corpus comparison | |

Progress and reasoning: [tasks](docs/TASKS.md), [decisions](docs/DECISIONS.md), [results](docs/RESULTS.md).

Every setting the paper does not publish is our own choice, and is recorded in `configs/`.

## Layout

| Path | Contents | Tracked in git |
|---|---|---|
| `configs/` | One YAML per run, plus sandbox image IDs | yes |
| `third_party/Table-Specialist/` | Microsoft repo, pinned at `af1921f` | submodule |
| `ts/corpus/` | Sample, filter and de-duplicate training tables | yes |
| `ts/bench/` | Rebuild test sets as `data.csv` + `info.json` | yes |
| `ts/llm/` | vLLM client with retries and a response cache | yes |
| `ts/gen/` | Question, SQL and Pandas generation | yes |
| `ts/validate/` | SQL-vs-Pandas agreement | yes |
| `ts/train/` | QLoRA trainer | yes |
| `ts/eval/` | Repo evaluators plus the corrected comparator | yes |
| `ts/loop.py` | Rounds, arms, checkpoint and resume | yes |
| `scripts/` | Setup and download scripts | yes |
| `data/`, `models/`, `runs/`, `logs/` | Downloads and outputs | no |

## Where it runs

Ubuntu 24.04 in WSL2, with its disk at `D:\WSL\Ubuntu`. Keep everything inside the Linux filesystem,
because reading Windows drives from WSL is much slower. GPU 0 serves the model and GPU 1 trains, because
GPU 1 also drives the display and has about 1.75 GB less free memory.

## Generated code

Model-written Scala runs only inside the repo's Docker sandbox, never on the host. Model-written SQL runs on the
host, read-only and SELECT only, with a 5-second limit.
