# Table-Specialist reproduction

A reproduction of the Llama-3.1-8B results from
[Table-LLM-Specialist](https://arxiv.org/abs/2410.12164) (EMNLP 2025), run on one workstation as a course project
at IIT Jammu, plus two extensions that try to recover the training data the paper's validator throws away.

**[Read the full report (PDF)](report/main.pdf)** · [End-semester slides (PDF)](presentation/endsem.pdf)

## What the paper does

Table-Specialist fine-tunes a model for one table task without human labels. For NL-to-SQL, the model writes a SQL
program and a Spark-Scala program for the same question. Only pairs that return the same rows on samples of the table
become training data, and the fine-tuned model generates the next round's data, for up to three rounds.

## Findings

- **The README's base checkpoint does not reproduce the paper.** Under every setting tried, Llama-3.1-8B base misses
  the paper's vanilla scores (WikiSQL 0.326 against 0.525).
- **The Instruct checkpoint does.** Through the repository's own chat call, Llama-3.1-8B-Instruct matches the paper's
  vanilla scores within 0.015 on WikiSQL, WikiTQ and BIRD. After three rounds of Table-Specialist, its zero-shot mean
  over four test sets is 0.6113 ± 0.0135, against the paper's 0.5983. Spider is the exception: every model here,
  vanilla included, scores 0.055–0.121 above the paper.
- **Validation pays off over rounds.** On base, Table-Specialist beats vanilla by +0.0647 and fine-tuning without
  validation by +0.0492 at round 3 (paired bootstrap, p < 0.0001, every seed positive). Unvalidated training wins at
  round 1 and gets worse every round after.
- **Preference learning hurts.** DPO on validated against rejected programs lowers accuracy (−0.0209, p < 0.0001).
- **Repairing rejected pairs helps some settings.** Repair-TS fixes each rejected pair once from its own execution
  feedback, then re-validates it with the paper's unchanged check. It gains +0.0421 on base NL-to-SQL at round 3
  (p < 0.0001, all seeds); a same-data control shows that about half of that gain is not data volume. It gains nothing
  on Instruct NL-to-SQL, and it helps NL-to-Scala and NL-to-R.

The report separates measurements from interpretations, grades the evidence for each claimed contribution and lists
every setting the paper does not publish. `report/tools/build_tables.py` generates every table in it from the files
in `results/`.

## What this repository adds to the original code

- The SQL-versus-Scala agreement check, which the released code does not include
- An NL-to-R validator (SQL against R)
- Rebuilt test sets for WikiSQL, Spider, WikiTQ and BIRD
- A corrected comparator for the scoring bug described in [`docs/DECISIONS.md`](docs/DECISIONS.md)
- A resumable loop over arms, seeds and rounds

## Layout

| Path | Contents |
|---|---|
| `report/` | LaTeX report, generated tables and the built PDF |
| `presentation/` | Beamer slides and the built PDF |
| `docs/` | [Decisions](docs/DECISIONS.md), [tasks](docs/TASKS.md) and a dated [lab log of results](docs/RESULTS.md) |
| `configs/` | Run settings, pinned requirements and sandbox image digests |
| `third_party/Table-Specialist/` | Microsoft repository, pinned at `af1921f` (submodule) |
| `ts/corpus/`, `ts/bench/` | Training-table corpus and test-set builders |
| `ts/gen/`, `ts/validate/` | Question and program generation, and the agreement checks |
| `ts/train/`, `ts/eval/` | QLoRA and DPO training, and evaluators |
| `ts/loop.py` and siblings | Rounds, arms, repair and resume |
| `results/` | Result summaries and case ID lists |
| `tests/` | Unit tests (pytest) |

Data, models, runs and logs stay out of git.

## Reproduce

The run needs Linux with two 24 GB GPUs, Docker and Hugging Face access to both Llama checkpoints. The full command
sequence, in the order it was run, is in Appendix H of the report (`report/appendices/H-reproduce.tex`). Setup:

```bash
git clone --recurse-submodules https://github.com/tajammulbasheer/table_specialist_repair
cd table_specialist_repair
python3.12 -m venv .venv && .venv/bin/pip install -r configs/requirements.lock
scripts/download_data.sh
.venv/bin/python -m pytest tests
```

The original runs used Ubuntu 24.04 under WSL2: GPU 0 served the model and GPU 1 trained it.

Model-written Scala runs only inside the repository's Docker sandbox. Model-written SQL runs on the host, read-only
and SELECT only, with a 5-second limit.

## License

MIT for the code in this repository. See [LICENSE](LICENSE). `third_party/Table-Specialist` keeps its own license.
