# Results

Last updated: 2026-11-04 12:10 (machine clock).

**Summary:** the faithful model is Llama-3.1-8B base. It meets the gate only on BIRD (0.364 against 0.388), and misses WikiSQL (0.326 against 0.525), WikiTQ (0.219 against 0.300) and, narrowly, Spider (0.725 against 0.670). The repo's few-shot prompt meets the gate on WikiTQ and BIRD but overshoots WikiSQL (0.685) and Spider (0.769). No setting tested reproduces the paper, and the gaps are recorded as unexplained. LoRA and Spark-Scala both work on this machine, and Spark makes validation take about 11 hours per round.

"Vanilla" means not fine-tuned. All Instruct scores below are a future-experiment reference, not the reproduction.

Raw summaries are in `results/`. "Repo-faithful" uses the authors' comparison, bug included. "Corrected" fixes the defects listed in `docs/DECISIONS.md`. Intervals are 95% Wilson intervals. No fine-tuned model has been trained yet.

## Targets from the paper

[Table 3](https://arxiv.org/abs/2410.12164), Llama3.1-8B, execution accuracy. The paper does not state which round the Table-Specialist column comes from.

| Benchmark | Cases | Vanilla | Table-Specialist |
|---|---|---|---|
| WikiSQL | 1,000 | 0.525 | 0.816 |
| WikiTQ | 1,000 | 0.300 | 0.449 |
| Text2Analysis | 271 | 0.273 | 0.465 |
| Spider | 1,198 | 0.670 | 0.690 |
| BIRD | 356 | 0.388 | 0.438 |

## Baseline gate (vanilla within ± 0.05 of the paper)

**Not passed.** No setting meets the gate on both WikiSQL and Spider.

### Earlier settings comparison

These rows use the earlier WikiSQL set of 975 cases and 1,198 Spider cases.

| Model and setting | WikiSQL faithful (interval) | WikiSQL corrected | Spider faithful (interval) | Spider corrected |
|---|---|---|---|---|
| Paper | 0.525 | n/a | 0.670 | n/a |
| Instruct, chat, no system message | 0.552 (0.520-0.583) | 0.553 | 0.787 (0.763-0.809) | 0.787 |
| Instruct, chat, repo system message | 0.519 (0.488-0.550) | 0.520 | 0.791 (0.767-0.813) | 0.791 |
| Base, raw completions | 0.327 (0.298-0.357) | 0.328 | 0.725 (0.699-0.749) | 0.753 |

- **Base**, the faithful model, misses WikiSQL by 0.198 and Spider by 0.055.
- **Instruct**, for reference, is within 0.05 on WikiSQL and misses Spider by about 0.12.
- **Comparator bug:** it changed no Spider verdicts for Instruct. On WikiSQL, the one Instruct difference is the `"Álbum"` case.
- **Base Spider gap:** 34 verdicts differ, and every one sampled was a query with the right rows in a different order, usually because the model added ORDER BY.

### Faithful baseline: base on the final test sets

Repo's zero-shot prompt, raw completions, at most 128 output tokens. One of four benchmarks is within ± 0.05.

| Benchmark | Cases | Faithful (interval) | Corrected | Paper | Gap | Gate |
|---|---|---|---|---|---|---|
| WikiSQL | 1,000 | 0.326 (0.298-0.356) | 0.326 | 0.525 | -0.199 | Missed |
| Spider | 1,198 | 0.725 (0.699-0.749) | 0.753 | 0.670 | +0.055 | Missed narrowly |
| WikiTQ | 1,000 | 0.219 (0.194-0.246) | 0.219 | 0.300 | -0.081 | Missed |
| BIRD | 363 | 0.364 (0.316-0.414) | 0.364 | 0.388 | -0.024 | Met |

On WikiSQL, 507 of 1,000 answers are SQL that fails to run, often from unquoted multi-word column names such as `WHERE Last Meeting = ...`.

### Diagnostic: base with the repo's few-shot prompt

Same cases, with the prompt from `nl2sql_generation_v6.py`. The paper's Appendix A shows this prompt but does not say whether evaluation used it. Two of four benchmarks are within ± 0.05, against one for zero-shot, and **neither prompt reproduces the paper on all four**.

| Benchmark | Zero-shot | Few-shot (interval) | Few-shot corrected | Paper | Few-shot gap | SQL that failed to run: zero-shot, few-shot |
|---|---|---|---|---|---|---|
| WikiSQL | 0.326 | 0.685 (0.656-0.713) | 0.685 | 0.525 | +0.160 | 507, 25 |
| Spider | 0.725 | 0.769 (0.744-0.792) | 0.772 | 0.670 | +0.099 | 20, 5 |
| WikiTQ | 0.219 | 0.344 (0.315-0.374) | 0.344 | 0.300 | +0.044 | 391, 16 |
| BIRD | 0.364 | 0.399 (0.350-0.451) | 0.399 | 0.388 | +0.011 | 41, 12 |

The paper's WikiSQL score falls between the two prompts. Few-shot fixes the quoting failures. 135 of its 315 WikiSQL misses contain a capitalised value that cannot match the lowercase table.

### Diagnostic: WikiSQL with original-case tables in the prompt

SQL still runs on the lowercase database. The idea was that the authors' prompt might show original case, because the repo keeps prompt tables apart from its databases.

| Prompt | Lowercase table | Original-case table | Paper |
|---|---|---|---|
| Zero-shot | 0.326 | 0.142 (0.122-0.165) | 0.525 |
| Few-shot | 0.685 | 0.287 (0.260-0.316) | 0.525 |

**This does not explain the gap**: no combination is near 0.525, and original case moves scores far below it. If capitalised values in the few-shot original-case answers were lowercased before running, the score would be 0.760, which shows how much value case matters.

**The WikiSQL gap is recorded as unexplained.** We stopped testing settings here, because trying more combinations would eventually land near the paper's score by chance. A question has gone into the draft author email.

### Future experiment: Instruct on the final test sets

Chat, with the repo's system message. Not part of the reproduction.

| Benchmark | Cases | Faithful (interval) | Corrected | Paper | Gate |
|---|---|---|---|---|---|
| WikiSQL | 1,000 | 0.518 (0.487-0.549) | 0.518 | 0.525 | Off by 0.007 |
| Spider | 1,198 | 0.791 (0.767-0.813) | 0.791 | 0.670 | Off by 0.121 |
| WikiTQ | 1,000 | Running | Running | 0.300 | Running |
| BIRD | 363 | Running | Running | 0.388 | Running |

## Where the misses come from

Instruct with no system message, for reference. Base will replace this table once its final runs finish.

| | WikiSQL (975) | Spider (1,198) |
|---|---|---|
| Correct | 538 | 943 |
| SQL failed to run | 323 | 26 |
| Right rows plus extra columns ("superset") | 4 | 82 |
| Wrong rows or values ("others", "subset") | 110 | 147 |

## Test-set construction

| Benchmark | Source questions | Cases | Gold SQL check |
|---|---|---|---|
| WikiSQL | 15,878 | 1,000; 16 skipped over the prompt limit | 0 mismatches against WikiSQL's engine |
| Spider | 2,147 | 1,230 single-table; 1,198 within the limit | 0 mismatches; 6 differ only in row order |
| WikiTQ | 4,344 | 1,000; 64 skipped over the limit | Gold is an answer list |
| BIRD | 1,534 | 363 single-table | 0 mismatches against the source databases |

## Speed and memory

Files in `results/smoke/`.

| Training | Blueprint assumed | Instruct, QLoRA, 1,500 tokens | Instruct, LoRA, 1,500 tokens | Instruct, LoRA, 4,000 tokens | **Base, LoRA, 1,500 tokens** | **Base, LoRA, 4,000 tokens** |
|---|---|---|---|---|---|---|
| Speed (tokens/s) | 1,500 (QLoRA) | 1,155 | 1,176 | 1,040 | 1,220 | 1,051 |
| Time (s/step) | Not stated | 1.30 | 1.28 | 3.85 | 1.23 | 3.81 |
| Peak VRAM reserved (GB) | 10-14 (QLoRA), 20-23 (LoRA) | 7.2 | 17.9 | 20.0 | 17.9 | 19.8 |

**LoRA on base fits on GPU 1**, which has about 23.3 GB free, at the study's longest sequences, with about 3.5 GB to spare.

| Serving | Blueprint assumed | Instruct | **Base** |
|---|---|---|---|
| Prefill (tokens/s) | 3,500 | 7,427 | 7,417 |
| Decode (tokens/s) | 800 | 262 | 262 |
| Model load (s) | Not stated | 167 | 92 |

| Spark sandbox, one small program | Measured |
|---|---|
| Programs that succeeded | 48 of 48, after raising the open-file limit (0 of 32 before) |
| Mean time per program, 24 workers, 1 CPU each | 39.7 s (the Blueprint assumed 0.8 s for Pandas) |
| Mean time per program, 36 workers, 1 CPU each | 46.0 s |
| Throughput | 2,133 programs per hour at 24 workers; 2,714 at 36 |
| Time for 30,000 programs (one round) | 14.1 hours at 24 workers; 11.1 at 36 |

More workers help only partly, because each program slows down as they are added. Real generated programs may be slower than this small test program. These runs shared the machine with GPU jobs.

## Loop test on 300 tables: first attempt, before the validator fixes

Table-Specialist arm, seed 1, 2 rounds. **These kept counts are superseded**, because validation had the bugs listed below. Files: `runs/old_test300_before_comparator_fix`, `logs/loop_test300_before_comparator_fix.log`.

| | Round 1 | Round 2 |
|---|---|---|
| Questions parsed, of 300 | 291 | 290 |
| SQL with a complete code block | 284 | 288 |
| Scala with a complete code block | 285 | 281 |
| Scala responses reaching the 256-token limit | 160 of 291 | 9 of 290 |
| **Kept** | **42** | **31** |
| Rejected: disagree | 95 | 114 |
| Rejected: Scala failed | 85 | 75 |
| Rejected: always empty | 48 | 45 |
| Rejected: no code | 13 | 11 |
| Rejected: SQL failed | 8 | 13 |
| Rejected: error while reading Spark's output | 0 | 1 |
| Scala runs | 631 | 516 |
| Generation / validation / training (s) | 372 / 1,002 / 322 | 416 / 887 / 261 |
| Peak VRAM in training (GB) | 18.6 | 19.1 |

**Round 2's limit count fell from 160 to 9** because the adapters learned to stop: training adds an end-of-text token, which vanilla does not produce after its code block.

### What inspecting these pairs found

We read 50 kept and 24 rejected pairs, and re-ran Scala for all 160 `scala_failed` pairs to get the error text.

| Finding | Evidence | Action |
|---|---|---|
| SQL ran on all-text columns; the repo reads each sample back with type inference | `execute.py` line 63 | Fixed |
| A one-column NULL result was counted as a disagreement | `gt_05606`: SQLite `[[null]]`, Spark file holds only its header. `gt_02180`: same with an empty string | Fixed |
| Different adapters would share cached answers under the names `sql` and `scala` | `ts/llm/client.py` key uses the name | Fixed before any second arm ran |
| `scala_failed` is not hiding false rejections | 160 of 160 exit with code 2. 89 name a column that does not resolve; 51 call a method the type lacks, such as `.sum()` on a DataFrame | No change |
| SQLite accepts an unknown quoted column as a string, so such SQL "succeeds" and returns the column's name | `gt_07112` returns `LenCM` per row | No change: Spark rejects these, so validation removes them. The no-validation arm keeps them, which is the paper's comparison |

## Loop test on 300 tables: corrected run

Table-Specialist arm, seed 1, 2 rounds. Files: `runs/train_tablespecialist_base_seed1_test300`, `logs/loop_test300.log`. **A pipeline test, not a result:** one seed and about 14% of the examples that a 3,000-table round should give.

| | Round 1 | Round 2 |
|---|---|---|
| Kept, of 300 tables | 41 | 43 |
| Rejected: disagree | 90 | 86 |
| Rejected: Scala failed | 79 | 79 |
| Rejected: always empty | 60 | 69 |
| Rejected: no code | 13 | 6 |
| Rejected: no question | 9 | 10 |
| Rejected: SQL failed | 8 | 7 |
| Scala runs | 615 | 613 |
| Generation / validation / training / evaluation (s) | 62 (cached) / 977 / 314 / 2,552 | 283 / 990 / 356 / 2,490 |

Accuracy, repo's scoring, vanilla and then the adapter of each round:

| Test set | Zero-shot: vanilla, R1, R2 | Few-shot: vanilla, R1, R2 |
|---|---|---|
| WikiSQL | 0.326, 0.533, 0.535 | 0.685, 0.532, 0.579 |
| Spider | 0.725, 0.710, 0.714 | 0.769, 0.690, 0.726 |
| WikiTQ | 0.219, 0.228, 0.259 | 0.344, 0.282, 0.342 |
| BIRD | 0.364, 0.322, 0.344 | 0.399, 0.344, 0.402 |

### Hand inspection: 50 kept and 50 rejected pairs

A random sample with seed 7, from both rounds. Verdicts come from reading the question, SQL and Scala **without opening each table**, so they judge whether the logic fits the question, and cannot confirm column contents.

| 50 kept pairs | Count | Numbers in the sample |
|---|---|---|
| SQL fits the question | 39 | |
| Doubtful (vague question, or logic cannot be judged from the text) | 5 | 8, 13, 20, 28, 44 |
| SQL does not answer the question | 6 | 2, 3, 9, 29, 33, 41; for example "How many" answered with a list |

| 50 rejected pairs | Count | Numbers in the sample |
|---|---|---|
| At least one side wrong: the rejection is right | 33 | |
| Same logic; differs only in output columns or in how ties are cut | 6 | 5, 14, 19, 33, 36, 43 |
| Logic looks right, but no row matches on any sample | 5 | 10, 16, 26, 29, 35 |
| SQL looks right, Scala program broken | 5 | 9, 21, 23, 31, 32 |
| Same logic, cause of disagreement not established | 1 | 22 |

So about 8 in 10 kept pairs look right, with 6 of 50 clearly wrong, and about 1 in 3 rejected pairs may have held usable SQL. No validator bug came out of this sample. The always-empty rule is ours: read literally, Algorithm 3 would keep a pair whose two results are both empty.

## Why pairs are rejected: corrected run, round 1

300 tables, 41 kept (13.7%). Sub-counts come from automatic rules plus re-running the first disagreeing sample, not from reading each pair. Files: `runs/train_tablespecialist_base_seed1_test300/round1/validated.jsonl`.

| Stage | Pairs | Breakdown |
|---|---|---|
| No usable question | 9 | |
| No complete code block | 13 | |
| SQL failed | 8 | |
| SQL returned nothing on all samples | 60 | 29 name a column not in the table (SQLite reads it as text); 4 use a filter value not in the table; 27 other |
| Scala failed | 79 | 28 also have SQL naming a column not in the table |
| SQL and Scala disagree | 90 | 36 different column count; 30 Scala empty but SQL has rows; 13 same shape, different values; 8 different row count; 3 SQL empty but Scala has rows |
| **Kept** | **41** | |

Some of the 36 column-count cases have the same filter and differ only in extra output columns (`gt_09111`, `gt_08622`, `gt_04286`). They have not been counted. If all 36 were kept, the total would be 77 of 300 (25.7%), so comparison strictness cannot explain most rejections.

The paper gives no pass rate or kept-pair count. Its validation discards failed pairs without repair or retry. These statements about the paper come from automated extraction of its HTML, and exact wording must be checked before citing it. Table 11 (GPT-3.5 only; not read directly) reports NL-to-SQL 0.594 with execution validation and 0.595 with a language-model validator, against vanilla 0.569.

## Core study: results so far

Scores below use repo scoring and come from seed 1, round 1 only; 2 more rounds and 2 more seeds are to come. Do not read them as study results yet. Source: `ROUND` lines in `logs/study_<arm>_seed1.log`.

| Benchmark, prompt | Vanilla base | Table-Specialist | No-validation | Paper vanilla / TS |
|---|---|---|---|---|
| WikiSQL, zero-shot | 0.326 | 0.680 | 0.710 | 0.525 / 0.816 |
| WikiSQL, few-shot | 0.685 | 0.649 | 0.680 | |
| Spider, zero-shot | 0.7245 | 0.7154 | 0.7604 | 0.670 / 0.690 |
| Spider, few-shot | 0.7688 | 0.7037 | 0.7262 | |
| WikiTQ, zero-shot | 0.219 | 0.292 | 0.351 | 0.300 / 0.449 |
| WikiTQ, few-shot | 0.344 | 0.312 | 0.333 | |
| BIRD, zero-shot | 0.3636 | 0.3664 | 0.3912 | 0.388 / 0.438 |
| BIRD, few-shot | 0.3994 | 0.3664 | 0.3829 | |

The paper does not state which prompt its scores used, so they sit on the zero-shot rows only for layout.

| Round 1 | Table-Specialist | No-validation |
|---|---|---|
| Questions generated from 3,000 tables | 2,904 | 2,901 |
| Pairs kept for training | 426 (14.7%) | 2,797 (96.4%) |
| Generation | 2,635 s | 2,638 s |
| Validation | 8,127 s | None |
| Training, both adapters | 3,044 s | 21,098 s |
| Evaluation, 8 runs | 2,654 s | 4,559 s |
| **Round total** | **16,460 s (4.57 h)** | **28,295 s (7.86 h)** |

Table-Specialist round 1 rejections, of 2,904: 855 Scala failed, 840 results disagree, 621 always empty, 105 no code, 53 SQL failed, 4 unexpected errors (`OverflowError`: an integer too large for SQLite). Table-Specialist round 2 kept 608 pairs of 2,918.

No-validation beat Table-Specialist in all 8 round 1 comparisons, while training on 6.6 times as many examples. The two arms' round 1 questions may also differ, as they ran concurrently. Evaluation took 72% longer in the no-validation arm; the cause has not been checked.

Failure on 2026-11-05: the no-validation arm stopped at 07:48 IST (02:18 UTC), when round 2 SQL training started. The training file reader used `str.splitlines`, which also breaks at U+2028, and 2 of 2,863 rows contained that character. Fixed in all 3 readers, with a regression test; `tests/test_loop.py` and `tests/test_phase4.py` pass (64). The arm restarted from its saved round 2 data on the same day, so only the failed start was repeated. Table-Specialist was not affected; its round 1 and round 2 training files hold no such character.

### Table-Specialist, seed 1, round 2 (finished 2026-11-05, 09:34 IST)

Kept 608 of 2,918 pairs (20.8%). Rejections: 952 results disagree, 658 Scala failed, 608 always empty, 59 SQL failed, 32 no code, 1 unexpected error. Time: 2,565 s generation, 10,408 s validation, 4,479 s training, 2,688 s evaluation; 20,140 s (5.59 h) in total. Scala runs: 7,926.

| Benchmark, prompt | Round 1 | Round 2 |
|---|---|---|
| WikiSQL, zero-shot | 0.680 | 0.756 |
| WikiSQL, few-shot | 0.649 | 0.750 |
| Spider, zero-shot | 0.7154 | 0.7287 |
| Spider, few-shot | 0.7037 | 0.7237 |
| WikiTQ, zero-shot | 0.292 | 0.297 |
| WikiTQ, few-shot | 0.312 | 0.325 |
| BIRD, zero-shot | 0.3664 | 0.3774 |
| BIRD, few-shot | 0.3664 | 0.3609 |

Known defect: after the restart, no-validation's `round1/summary.json` shows zero timings, because a resumed round rewrites its summary. The original timings are in the first `ROUND` line of `logs/study_no_validation_seed1.log` and in the table above.

### Table-Specialist, seed 1, all 3 rounds (finished 2026-11-05, 17:35 IST)

One seed only, repo scoring. Pairs kept: 426 (14.7%), 608 (20.8%), 672 (23.0%). Round 3's SQL adapter was resumed from `checkpoint-72` after the crash, so its summary has `"resumed": true`: its timings and SQL `train_loss` cover only the work after the restart. Evaluation took 2,434 s.

| Benchmark, prompt | Vanilla | Round 1 | Round 2 | Round 3 | Paper vanilla / TS |
|---|---|---|---|---|---|
| WikiSQL, zero-shot | 0.326 | 0.680 | 0.756 | 0.790 | 0.525 / 0.816 |
| WikiSQL, few-shot | 0.685 | 0.649 | 0.750 | 0.772 | |
| Spider, zero-shot | 0.7245 | 0.7154 | 0.7287 | 0.7496 | 0.670 / 0.690 |
| Spider, few-shot | 0.7688 | 0.7037 | 0.7237 | 0.7337 | |
| WikiTQ, zero-shot | 0.219 | 0.292 | 0.297 | 0.354 | 0.300 / 0.449 |
| WikiTQ, few-shot | 0.344 | 0.312 | 0.325 | 0.338 | |
| BIRD, zero-shot | 0.3636 | 0.3664 | 0.3774 | 0.405 | 0.388 / 0.438 |
| BIRD, few-shot | 0.3994 | 0.3664 | 0.3609 | 0.3857 | |

The paper does not state which prompt its scores used, so they sit on the zero-shot rows only for layout.

### Seed 1, round 2: the two arms compared

One seed only, repo scoring. `no_validation` finished at 16:51 IST on 2026-11-05 and kept 2,863 of 2,918 pairs; `table_specialist` kept 608. The round's SQL adapter was finished before the crash, so the crash did not touch these scores. Its summary has `"resumed": true`: evaluation took 3,106 s, while its Scala training time of 807 s and generation time cover only the work after the restart.

| Benchmark, prompt | Table-Specialist | No-validation | Difference (TS minus no-validation) |
|---|---|---|---|
| WikiSQL, zero-shot | 0.756 | 0.676 | +0.080 |
| WikiSQL, few-shot | 0.750 | 0.639 | +0.111 |
| Spider, zero-shot | 0.7287 | 0.7237 | +0.0050 |
| Spider, few-shot | 0.7237 | 0.6953 | +0.0284 |
| WikiTQ, zero-shot | 0.297 | 0.350 | -0.053 |
| WikiTQ, few-shot | 0.325 | 0.325 | 0.000 |
| BIRD, zero-shot | 0.3774 | 0.3939 | -0.0165 |
| BIRD, few-shot | 0.3609 | 0.3857 | -0.0248 |

Table-Specialist is higher on 4, no-validation on 3, with 1 tie. From round 1 to round 2, no-validation fell on WikiSQL (0.710 to 0.676, 0.680 to 0.639) and Spider (0.7604 to 0.7237, 0.7262 to 0.6953), and Table-Specialist rose on both.

### Table-Specialist, all 3 seeds, round 3 (finished 2026-11-07, 03:12 IST)

The arm's final result: the round 3 adapters of seeds 1, 2 and 3, kept 672, 777 and 729 pairs. Mean ± sample standard deviation over 3 seeds, repo scoring. Seed 3 round 3 kept 729 of 2,908 pairs (25.1%).

| Benchmark, prompt | Vanilla | TS mean ± std | TS − vanilla | Paper vanilla → TS |
|---|---|---|---|---|
| WikiSQL, zero-shot | 0.326 | 0.779 ± 0.010 | +0.453 | 0.525 → 0.816 |
| WikiSQL, few-shot | 0.685 | 0.768 ± 0.004 | +0.083 | |
| Spider, zero-shot | 0.7245 | 0.7031 ± 0.0405 | −0.0214 | 0.670 → 0.690 |
| Spider, few-shot | 0.7688 | 0.6881 ± 0.0400 | −0.0807 | |
| WikiTQ, zero-shot | 0.219 | 0.322 ± 0.034 | +0.103 | 0.300 → 0.449 |
| WikiTQ, few-shot | 0.344 | 0.330 ± 0.019 | −0.014 | |
| BIRD, zero-shot | 0.3636 | 0.3802 ± 0.0263 | +0.0166 | 0.388 → 0.438 |
| BIRD, few-shot | 0.3994 | 0.3774 ± 0.0099 | −0.0220 | |

Seed 3 round 3 scores: 0.775, 0.769, 0.6845, 0.6594, 0.325, 0.344, 0.3829, 0.3802 (same order as the table below).

### Both arms, every finished round (updated 2026-11-07, 03:20 IST)

Repo scoring. Order of scores: WikiSQL zero-shot, WikiSQL few-shot, Spider zero-shot, Spider few-shot, WikiTQ zero-shot, WikiTQ few-shot, BIRD zero-shot, BIRD few-shot. TS = Table-Specialist, NV = no-validation.

| Seed, round | TS kept | TS scores | NV kept | NV scores | TS higher / NV higher / tied |
|---|---|---|---|---|---|
| 1, 1 | 426 | 0.680, 0.649, 0.7154, 0.7037, 0.292, 0.312, 0.3664, 0.3664 | 2,797 | 0.710, 0.680, 0.7604, 0.7262, 0.351, 0.333, 0.3912, 0.3829 | 0 / 8 / 0 |
| 1, 2 | 608 | 0.756, 0.750, 0.7287, 0.7237, 0.297, 0.325, 0.3774, 0.3609 | 2,863 | 0.676, 0.639, 0.7237, 0.6953, 0.350, 0.325, 0.3939, 0.3857 | 4 / 3 / 1 |
| 1, 3 | 672 | 0.790, 0.772, 0.7496, 0.7337, 0.354, 0.338, 0.405, 0.3857 | 2,854 | 0.692, 0.627, 0.682, 0.6644, 0.330, 0.325, 0.3747, 0.3581 | 8 / 0 / 0 |
| 2, 1 | 388 | 0.704, 0.733, 0.6928, 0.6953, 0.269, 0.311, 0.3471, 0.3747 | 2,788 | 0.666, 0.665, 0.7462, 0.7137, 0.353, 0.331, 0.4132, 0.3857 | 2 / 6 / 0 |
| 2, 2 | 655 | 0.751, 0.749, 0.6803, 0.6811, 0.270, 0.300, 0.3196, 0.3361 | 2,852 | 0.638, 0.629, 0.702, 0.6669, 0.330, 0.304, 0.3857, 0.3691 | 3 / 5 / 0 |
| 2, 3 | 777 | 0.772, 0.764, 0.6753, 0.6711, 0.286, 0.308, 0.3526, 0.3664 | 2,832 | 0.675, 0.642, 0.6628, 0.6361, 0.308, 0.283, 0.3636, 0.3416 | 6 / 2 / 0 |
| 3, 1 | 423 | 0.785, 0.780, 0.7003, 0.6861, 0.342, 0.342, 0.3967, 0.3939 | 2,797 | 0.700, 0.707, 0.7538, 0.7254, 0.367, 0.337, 0.3939, 0.3691 | 5 / 3 / 0 |
| 3, 2 | 618 | 0.731, 0.724, 0.7062, 0.6953, 0.324, 0.331, 0.3857, 0.3829 | 2,819 | 0.666, 0.677, 0.7245, 0.7062, 0.339, 0.320, 0.4022, 0.3664 | 4 / 4 / 0 |
| 3, 3 | 729 | 0.775, 0.769, 0.6845, 0.6594, 0.325, 0.344, 0.3829, 0.3802 | 2,840 | 0.621, 0.582, 0.6953, 0.6661, 0.310, 0.304, 0.3774, 0.3416 | 6 / 2 / 0 |

Final tally (72 comparisons, 2026-11-07, 19:00 IST): TS higher 38, NV higher 33, tied 1. By round: round 1 7 / 17 / 0, round 2 11 / 12 / 1, round 3 20 / 4 / 0.

### Core study finished: both arms, 3 seeds (2026-11-07, 19:00 IST)

`no_validation` seed 3 exited 0 at 19:00 IST, completing 3 seeds × 3 rounds for both arms. Mean ± sample std over 3 seeds, repo scoring. Difference = TS − NV, then per seed (1, 2, 3).

**Round 3 (the final adapters):**

| Benchmark, prompt | Vanilla | TS | NV | TS − NV | Per seed |
|---|---|---|---|---|---|
| WikiSQL, zero-shot | 0.326 | 0.779 ± 0.010 | 0.663 ± 0.037 | +0.116 | +0.098, +0.097, +0.154 |
| WikiSQL, few-shot | 0.685 | 0.768 ± 0.004 | 0.617 ± 0.031 | +0.151 | +0.145, +0.122, +0.187 |
| Spider, zero-shot | 0.7245 | 0.7031 ± 0.0405 | 0.6800 ± 0.0163 | +0.0231 | +0.0676, +0.0125, −0.0108 |
| Spider, few-shot | 0.7688 | 0.6881 ± 0.0400 | 0.6555 ± 0.0169 | +0.0325 | +0.0693, +0.0350, −0.0067 |
| WikiTQ, zero-shot | 0.219 | 0.322 ± 0.034 | 0.316 ± 0.012 | +0.006 | +0.024, −0.022, +0.015 |
| WikiTQ, few-shot | 0.344 | 0.330 ± 0.019 | 0.304 ± 0.021 | +0.026 | +0.013, +0.025, +0.040 |
| BIRD, zero-shot | 0.3636 | 0.3802 ± 0.0263 | 0.3719 ± 0.0073 | +0.0083 | +0.0303, −0.0110, +0.0055 |
| BIRD, few-shot | 0.3994 | 0.3774 ± 0.0099 | 0.3471 ± 0.0095 | +0.0303 | +0.0276, +0.0248, +0.0386 |

TS mean is higher on all 8; it is higher in all 3 seeds on WikiSQL (both prompts), WikiTQ few-shot and BIRD few-shot.

**Mean TS − NV by round** (order as above):

| Round | WikiSQL z / f | Spider z / f | WikiTQ z / f | BIRD z / f |
|---|---|---|---|---|
| 1 | +0.031 / +0.037 | −0.0506 / −0.0267 | −0.056 / −0.012 | −0.0294 / −0.0009 |
| 2 | +0.086 / +0.093 | −0.0117 / +0.0106 | −0.043 / +0.002 | −0.0330 / −0.0138 |
| 3 | +0.116 / +0.151 | +0.0231 / +0.0325 | +0.006 / +0.026 | +0.0083 / +0.0303 |

NV round 3 means fall below its round 1 means on all 8 (for example WikiSQL zero-shot 0.692 → 0.663, Spider zero-shot 0.7535 → 0.6800); TS round 3 means are at or above its round 1 means on 7 of 8 (Spider few-shot 0.6950 → 0.6881 is the exception). Against vanilla, TS round 3 is still below on Spider (both prompts), WikiTQ few-shot and BIRD few-shot.

Seed 2 round 1 took 113 s of generation in no-validation: the questions and code came from the response cache, because both arms start each seed from vanilla on the same tables. Checked by table: 2,999 of 3,000 questions and 2,887 of 2,888 SQL and Scala programs are identical between the arms. So round 1 of seeds 2 and 3 is a near-paired comparison: the arms differ only in validation. Seed 1 round 1 ran both arms at the same moment, so its questions may differ.

### Table-Specialist, seed 2, round 1, against seed 1

First uninterrupted repeat, repo scoring. Seed 2 kept 388 of 2,888 pairs (13.4%), against 426 of 2,904 (14.7%) in seed 1. The gaps show how far one seed can move a score, so differences of this size between arms need all 3 seeds.

| Benchmark, prompt | Seed 1 | Seed 2 | Difference |
|---|---|---|---|
| WikiSQL, zero-shot | 0.680 | 0.704 | +0.024 |
| WikiSQL, few-shot | 0.649 | 0.733 | +0.084 |
| Spider, zero-shot | 0.7154 | 0.6928 | -0.0226 |
| Spider, few-shot | 0.7037 | 0.6953 | -0.0084 |
| WikiTQ, zero-shot | 0.292 | 0.269 | -0.023 |
| WikiTQ, few-shot | 0.312 | 0.311 | -0.001 |
| BIRD, zero-shot | 0.3664 | 0.3471 | -0.0193 |
| BIRD, few-shot | 0.3664 | 0.3747 | +0.0083 |

### Crash on 2026-11-05, about 08:15 UTC

Both arms stopped: `no_validation` round 2 Scala training at step 277 of 297 (`checkpoint-275`), and `table_specialist` round 3 SQL training at step 77 of 288 (`checkpoint-72`). Both checkpoints loaded on the CPU with no NaNs, and short resumed runs from copies of them on the GPUs exited 0. Table-Specialist round 3 validation had finished: kept 672 of 2,918 pairs (23.0%). Rejections: 900 results disagree, 622 always empty, 559 Scala failed, 95 SQL failed, 67 no code, 3 unexpected errors. Scala runs: 8,614.

Training speed per example: 1.16-1.26 s in finished runs across both arms, and 1.35 s in the interrupted `no_validation` Scala run, which overlapped Table-Specialist validation. The roughly 35 s per step in `no_validation` is normal, as it accumulates 29 examples per step against 7 in Table-Specialist.

## Extension 2: learning from rejected programs, round 1 (all 3 seeds, finished 2026-11-08, 09:59 IST)

Design in `docs/DECISIONS.md`. `bestof4` = SFT on one validated program per question from 4 candidates (greedy + 3 sampled); `dpo` = `bestof4` then DPO on (chosen, rejected) pairs. Same questions and tables as TS round 1. Repo scoring.

**Labelling:**

| Seed | Questions | With a validated program (TS / bestof4) | Recovered from samples | DPO pairs (rejected rows differ / always empty / SQL failed) | Scala runs | Labelling time |
|---|---|---|---|---|---|---|
| 1 | 2,904 | 426 / 673 | 247 (+58%) | 431 (312 / 95 / 24) | 7,329 | 3.2 h |
| 2 | 2,888 | 388 / 627 | 239 (+62%) | 438 (318 / 107 / 13) | 7,286 | 2.7 h over two runs |
| 3 | 2,893 | 423 / 643 | 220 (+52%) | 397 (302 / 68 / 27) | 7,147 | 2.7 h |

TS's own greedy program was `rejected` for 75 (seed 1), 89 (seed 2) and 73 (seed 3) questions: a sampled program validated, and the greedy SQL gives different rows.

**Scores** (order: WikiSQL z, f; Spider z, f; WikiTQ z, f; BIRD z, f):

| Seed | Model | Scores | Higher / lower than TS | Mean − TS |
|---|---|---|---|---|
| 1 | TS | 0.680, 0.649, 0.7154, 0.7037, 0.292, 0.312, 0.3664, 0.3664 | | |
| 1 | bestof4 | 0.667, 0.680, 0.6778, 0.6886, 0.313, 0.339, 0.3829, 0.3884 | 5 / 3 | +0.0065 |
| 1 | dpo | 0.685, 0.660, 0.6761, 0.6661, 0.232, 0.321, 0.3030, 0.3444 | 3 / 5 | −0.0247 |
| 2 | TS | 0.704, 0.733, 0.6928, 0.6953, 0.269, 0.311, 0.3471, 0.3747 | | |
| 2 | bestof4 | 0.759, 0.763, 0.7329, 0.7062, 0.299, 0.321, 0.3912, 0.3802 | **8 / 0** | **+0.0282** |
| 2 | dpo | 0.764, 0.776, 0.7170, 0.6995, 0.255, 0.290, 0.3691, 0.3499 | 5 / 3 | +0.0117 |
| 3 | TS | 0.785, 0.780, 0.7003, 0.6861, 0.342, 0.342, 0.3967, 0.3939 | | |
| 3 | bestof4 | 0.721, 0.735, 0.7104, 0.7003, 0.330, 0.339, 0.3857, 0.3802 | 2 / 6 | −0.0156 |
| 3 | dpo | 0.719, 0.651, 0.7012, 0.6694, 0.285, 0.292, 0.3554, 0.3554 | 1 / 7 | −0.0497 |

**Mean of 3 seeds, round 1** (same column order; "higher / lower" compares the 8 seed-mean columns):

| Model | Scores | Mean of 8 | Higher / lower than TS round 1 | Mean − TS round 1 |
|---|---|---|---|---|
| TS round 1 | 0.7230, 0.7207, 0.7028, 0.6950, 0.3010, 0.3217, 0.3701, 0.3783 | 0.5266 | | |
| NV round 1 | 0.6920, 0.6840, 0.7535, 0.7218, 0.3570, 0.3337, 0.3994, 0.3792 | 0.5401 | 6 / 2 | +0.0135 |
| bestof4 | 0.7157, 0.7260, 0.7070, 0.6984, 0.3140, 0.3330, 0.3866, 0.3829 | 0.5330 | **7 / 1** | +0.0064 |
| dpo | 0.7227, 0.6957, 0.6981, 0.6783, 0.2573, 0.3010, 0.3425, 0.3499 | 0.5057 | 0 / 8 | −0.0209 |
| TS round 3 (reference) | 0.7790, 0.7683, 0.7031, 0.6881, 0.3217, 0.3300, 0.3802, 0.3774 | 0.5435 | | |

- `bestof4` is above TS round 1 in 7 of 8 seed-mean columns, but by +0.0064 on average, and the per-seed result is not stable: +0.0282 (seed 2), +0.0065 (seed 1), −0.0156 (seed 3). It is below NV round 1 by 0.0071 (3 higher / 5 lower) and below TS round 3 by 0.0105 (5 / 3).
- `dpo` (first settings) is below `bestof4` in all 3 seeds: 1/7, 2/6, 0/8; mean −0.0273. In seed 3 the errors rise less (WikiTQ zero-shot 8 → 18) and most of the loss is wrong answers that run.

`dpo` against `bestof4`: 1 higher / 7 lower (mean −0.0311) in seed 1, 2 / 6 (−0.0165) in seed 2, 0 / 8 (−0.0342) in seed 3.

**Why DPO lowers scores:** it writes invalid SQL far more often. Compile or execution errors, zero-shot, `bestof4` → `dpo`: WikiTQ 15 → 58 (seed 1) and 17 → 55 (seed 2); BIRD 9 → 11 and 9 → 28. Training ended with reward margins of 3.40, 5.35 and 3.19 and reward accuracy 0.98-1.00 on 397-438 pairs over 2 epochs, so the policy moved far from the reference. The SFT term did not prevent it.

### bestof4 against every arm, test sets, mean ± std of 3 seeds (2026-11-08, 11:20 IST)

Repo scoring. bestof4 is a round 1 model, so TS round 1 and NV round 1 are the like-for-like comparisons; round 3 is shown as the final adapters of the core study. Std is the sample standard deviation over the 3 seeds.

| Benchmark, prompt | Vanilla | bestof4 | TS round 1 | NV round 1 | TS round 3 | NV round 3 | bestof4 per seed |
|---|---|---|---|---|---|---|---|
| WikiSQL, zero-shot | 0.326 | 0.7157 ± 0.0462 | 0.7230 ± 0.0550 | 0.6920 ± 0.0231 | 0.7790 ± 0.0096 | 0.6627 ± 0.0371 | 0.667, 0.759, 0.721 |
| WikiSQL, few-shot | 0.685 | 0.7260 ± 0.0422 | 0.7207 ± 0.0664 | 0.6840 ± 0.0213 | 0.7683 ± 0.0040 | 0.6170 ± 0.0312 | 0.680, 0.763, 0.735 |
| Spider, zero-shot | 0.7245 | 0.7070 ± 0.0277 | 0.7028 ± 0.0115 | 0.7535 ± 0.0071 | 0.7031 ± 0.0405 | 0.6800 ± 0.0163 | 0.6778, 0.7329, 0.7104 |
| Spider, few-shot | 0.7688 | 0.6984 ± 0.0090 | 0.6950 ± 0.0088 | 0.7218 ± 0.0070 | 0.6881 ± 0.0400 | 0.6555 ± 0.0169 | 0.6886, 0.7062, 0.7003 |
| WikiTQ, zero-shot | 0.219 | 0.3140 ± 0.0155 | 0.3010 ± 0.0373 | 0.3570 ± 0.0087 | 0.3217 ± 0.0341 | 0.3160 ± 0.0122 | 0.313, 0.299, 0.330 |
| WikiTQ, few-shot | 0.344 | 0.3330 ± 0.0104 | 0.3217 ± 0.0176 | 0.3337 ± 0.0031 | 0.3300 ± 0.0193 | 0.3040 ± 0.0210 | 0.339, 0.321, 0.339 |
| BIRD, zero-shot | 0.3636 | 0.3866 ± 0.0042 | 0.3701 ± 0.0250 | 0.3994 ± 0.0120 | 0.3802 ± 0.0263 | 0.3719 ± 0.0073 | 0.3829, 0.3912, 0.3857 |
| BIRD, few-shot | 0.3994 | 0.3829 ± 0.0047 | 0.3783 ± 0.0141 | 0.3792 ± 0.0089 | 0.3774 ± 0.0099 | 0.3471 ± 0.0095 | 0.3884, 0.3802, 0.3802 |
| **Mean of 8** | 0.4788 | 0.5330 | 0.5266 | 0.5401 | 0.5435 | 0.4943 | 0.5171, 0.5441, 0.5377 |

| bestof4 against | Seed-mean columns higher / lower (of 8) | Seed × column higher / lower (of 24) | Mean of 8 difference |
|---|---|---|---|
| Vanilla | 4 / 4 | | +0.0542 |
| TS round 1 | 7 / 1 | 15 / 9 | +0.0064 |
| NV round 1 | 3 / 5 | 8 / 15, 1 tied | −0.0071 |
| TS round 3 | 5 / 3 | 12 / 11, 1 tied | −0.0105 |
| NV round 3 | 7 / 1 | 20 / 4 | +0.0387 |

### DPO tuning on the dev sets, seed 1 (finished 2026-11-08, 11:04 IST)

Zero-shot, repo scoring, dev sets from `ts/bench/dev.py` (WikiSQL 300, Spider single-table 575, WikiTQ 300). Every variant trains DPO from seed 1's `bestof4` adapter. Selection rule fixed beforehand: test only if the best variant's dev mean is above `bestof4`'s.

| Model | WikiSQL | Spider | WikiTQ | Mean | − bestof4 | Pairs | Steps | Final reward margin |
|---|---|---|---|---|---|---|---|---|
| TS round 1 | 0.6900 | 0.7591 | 0.2733 | **0.5741** | +0.0123 | | | |
| bestof4 | 0.6667 | 0.7421 | 0.2767 | 0.5618 | 0 | | | |
| first DPO (lr 2e-5, beta 0.1, 2 epochs) | 0.6700 | 0.7129 | 0.2400 | 0.5410 | −0.0208 | 431 | | 3.40 |
| v1: lr 5e-6, 1 epoch | 0.6733 | 0.7324 | 0.2667 | 0.5575 | −0.0043 | 431 | 27 | 0.21 |
| v2: v1 + beta 0.5 | 0.6767 | 0.7299 | 0.2733 | 0.5600 | −0.0018 | 431 | 27 | 0.20 |
| v3: v1 + rows-differ negatives only | 0.6600 | 0.7397 | 0.2700 | 0.5566 | −0.0052 | 312 | 20 | 0.14 |
| v4: beta 0.5 | 0.6633 | 0.7056 | 0.2433 | 0.5374 | −0.0244 | 431 | 54 | 3.40 |
| v5: lr 1e-6, 2 epochs | 0.6567 | 0.7372 | 0.2733 | 0.5557 | −0.0061 | 431 | 54 | 0.09 |

- **Selection rule outcome: no variant is above `bestof4`'s dev mean (0.5618); the best is v2 at 0.5600. DPO is not run on the test sets again, and is reported as not helping.**
- Gentler DPO (v1, v2, v3, v5) removes most of the damage (−0.0208 → −0.0018 to −0.0061). Raising beta alone at lr 2e-5 (v4) does not: −0.0244, with the same final margin as the first DPO (3.40). The learning rate, not beta, drives the drop. The differences between v1, v2 and v3 (0.0034) are smaller than one question in 300 on WikiSQL/WikiTQ (0.0033), so they are noise.
- On dev, TS round 1 is above `bestof4` by 0.0123. This agrees with seed 1's test sets for the same 3 benchmarks, zero-shot: TS 0.5625, `bestof4` 0.5526 (TS +0.0099). `bestof4`'s +0.0065 over all 8 seed 1 test columns comes from WikiSQL few-shot, both WikiTQ columns and both BIRD columns.

## Extension 3: repair of rejected pairs, pilot (seed 1, 300 tables, finished 2026-11-08, 12:00 IST)

Design and go / no-go rule in `docs/DECISIONS.md`. Output `runs/repair_pilot_seed1_test300/round1`; 22 min on GPU 0 and 36 CPUs, 288 Scala runs.

| Rejected because | Repair attempts | Recovered |
|---|---|---|
| SQL and Scala disagree | 90 | 3 (all by repairing the Scala) |
| Scala failed | 79 | 3 |
| SQL always empty | 60 | 1 |
| SQL failed | 8 | 0 |
| No complete code block | 13 | not repaired |
| **Total** | **237** | **7** |

- Kept pairs 41 → 48: **+17.1%** (+14.6%, 6 of 41, without the 10 smoke-test pairs). The rule needed +30% (54 or more): **no-go**.
- Hand check of all 7 recovered pairs, each against its full table: **7 of 7 SQL labels answer the question correctly** (rule: 80%). 6 of 7 keep the original SQL: it was right, and only the Scala was repaired. So repair is precise but recovers little.
- Why so little: base Llama-3.1-8B copied the failed code back in 85 of 169 Scala repairs and 42 of 158 SQL repairs, even after 3 samples at temperature 0.7. 49 of 60 always-empty SQL stayed empty after repair; the question itself may have no matching rows, which code repair cannot fix.
- Expected effect at full scale: about +17% validated pairs per seed. `bestof4` added +52-62% and moved the test mean by only +0.0064, so a gain from +17% is unlikely to be measurable. Full repair runs are not started.

### Extension 3, second pilot: Llama-3.1-8B-Instruct as repairer (finished 2026-11-08, 12:30 IST)

Same 300 tables, same rejected pairs and diagnoses, same retries, Algorithm 3 and rule. Output `runs/repair_pilot_instruct_seed1_test300/round1`; 20 min, 591 Scala runs.

| Rejected because | Attempts | Recovered: base repairer | Recovered: Instruct repairer |
|---|---|---|---|
| SQL and Scala disagree | 90 | 3 | 11 |
| Scala failed | 79 | 3 | 9 |
| SQL always empty | 60 | 1 | 3 |
| SQL failed | 8 | 0 | 0 |
| **Total** | **237** | **7** | **23** |
| Kept pairs (first pass 41) | | 48 (+17.1%) | **64 (+56.1%)** |

- Instruct wrote new code in 314 of 327 repairs (base: 200 of 327). It recovered 6 of base's 7 pairs; together the two recover 24.
- None of the 23 are among the 10 smoke-test pairs.
- 18 of 23 recovered by repairing the Scala, so the SQL label is TS's original SQL; 5 have a repaired SQL.
- **Hand check, all 23, each against its full table: 21 correct, 2 doubtful, 0 wrong (91.3%)**. Doubtful: `gt_08669` (the MAX of random string IDs; the question is ill-posed) and `gt_02705` ("average dwarf star mass" answered with the column `mdeg/keV`; the only numeric reading of a vague question). Examples of SQL that repair fixed: `gt_06535` `COUNT("Class count")` → `"Class count"` (70); `gt_02813` `"Finish" = 6` → `'6th'`.
- **Outcome: go** (+56.1% ≥ +30%; 91.3% ≥ 80%; also 21 of 23 if both doubtful are counted wrong).

### Extension 3, full run round 1, seeds 1-2 (seed 2 finished 2026-11-08, 19:44 IST, exit 0; seed 3 running)

Repair of the pairs TS round 1 rejected, Instruct repairer:

| Seed | Rejected, tried | Recovered | Disagree | Scala failed | Always empty | SQL failed | Kept: TS → repair-TS |
|---|---|---|---|---|---|---|---|
| 1 | 2,369 | 224 | 159 of 840 | 55 of 855 | 9 of 621 | 1 of 53 | 426 → 650 (+52.6%) |
| 2 | 2,398 | 217 | 146 of 820 | 58 of 857 | 13 of 660 | 0 of 61 | 388 → 605 (+55.9%) |

Seed 1 by change: new Scala 169, new SQL 36, both 19 (5,465 Scala runs). Seed 2: new Scala 172, new SQL 34, both 11.

Test, repo-faithful accuracy (z zero-shot, f few-shot). Vanilla is greedy and the same for every seed.

| Seed | Arm | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 | Repair-TS higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | Vanilla | 0.3260 | 0.6850 | 0.7245 | 0.7688 | 0.2190 | 0.3440 | 0.3636 | 0.3994 | 0.4788 | s1 4/4, s2 5/2 (1 tie) |
| 1 | TS round 1 | 0.6800 | 0.6490 | 0.7154 | 0.7037 | 0.2920 | 0.3120 | 0.3664 | 0.3664 | 0.5106 | 6/2 |
| 1 | bestof4 round 1 | 0.6670 | 0.6800 | 0.6778 | 0.6886 | 0.3130 | 0.3390 | 0.3829 | 0.3884 | 0.5171 | 5/2 (1 tie) |
| 1 | NV round 1 | 0.7100 | 0.6800 | 0.7604 | 0.7262 | 0.3510 | 0.3330 | 0.3912 | 0.3829 | 0.5418 | 4/4 |
| 1 | TS round 3 | 0.7900 | 0.7720 | 0.7496 | 0.7337 | 0.3540 | 0.3380 | 0.4050 | 0.3857 | 0.5660 | 1/7 |
| 1 | **Repair-TS round 1** | 0.7350 | 0.7350 | 0.7045 | 0.6945 | 0.3430 | 0.3350 | 0.3747 | 0.3884 | **0.5388** | |
| 2 | TS round 1 | 0.7040 | 0.7330 | 0.6928 | 0.6953 | 0.2690 | 0.3110 | 0.3471 | 0.3747 | 0.5159 | 8/0 |
| 2 | bestof4 round 1 | 0.7590 | 0.7630 | 0.7329 | 0.7062 | 0.2990 | 0.3210 | 0.3912 | 0.3802 | 0.5441 | 6/1 (1 tie) |
| 2 | NV round 1 | 0.6660 | 0.6650 | 0.7462 | 0.7137 | 0.3530 | 0.3310 | 0.4132 | 0.3857 | 0.5342 | 5/3 |
| 2 | TS round 3 | 0.7720 | 0.7640 | 0.6753 | 0.6711 | 0.2860 | 0.3080 | 0.3526 | 0.3664 | 0.5244 | 7/1 |
| 2 | **Repair-TS round 1** | 0.7820 | 0.7590 | 0.7337 | 0.7220 | 0.3370 | 0.3430 | 0.3912 | 0.3994 | **0.5584** | |

Mean of 8 cells, mean of seeds 1-2: repair-TS 0.5486; TS round 1 0.5133 (+0.0354); bestof4 0.5306 (+0.0180); NV round 1 0.5380 (+0.0106); TS round 3 0.5452 (+0.0034); vanilla 0.4788 (+0.0698). Repair-TS is above TS round 1 on both seeds (+0.0282, +0.0425); on seed 1 it is below NV round 1 (−0.0031) and TS round 3 (−0.0272).

All round 1 arms with DPO (first DPO; seed 1 0.4859, seed 2 0.5276), mean of seeds 1-2 per cell:

| Arm | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 0.3260 | 0.6850 | 0.7245 | **0.7688** | 0.2190 | **0.3440** | 0.3636 | **0.3994** | 0.4788 |
| NV | 0.6880 | 0.6725 | **0.7533** | 0.7199 | **0.3520** | 0.3320 | **0.4022** | 0.3843 | 0.5380 |
| TS | 0.6920 | 0.6910 | 0.7041 | 0.6995 | 0.2805 | 0.3115 | 0.3568 | 0.3705 | 0.5132 |
| bestof4 | 0.7130 | 0.7215 | 0.7053 | 0.6974 | 0.3060 | 0.3300 | 0.3871 | 0.3843 | 0.5306 |
| DPO | 0.7245 | 0.7180 | 0.6966 | 0.6828 | 0.2435 | 0.3055 | 0.3360 | 0.3471 | 0.5068 |
| Repair-TS | **0.7585** | **0.7470** | 0.7191 | 0.7083 | 0.3400 | 0.3390 | 0.3830 | 0.3939 | **0.5486** |

### Extension 3, full run round 1, all 3 seeds (seed 3 finished 2026-11-08, 22:16 IST, exit 0)

Seed 3 repair: 204 of 2,370 recovered (disagree 143 of 781, Scala failed 54 of 876, always empty 7 of 656, SQL failed 0 of 57); kept 423 → 627 (+48.2%); new Scala 147, new SQL 35, both 22; 5,230 Scala runs. Over 3 seeds: 645 of 7,137 recovered, kept 1,237 → 1,882 (+52.1%).

Seed 3 test:

| Arm | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 | Repair-TS higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 0.3260 | 0.6850 | 0.7245 | 0.7688 | 0.2190 | 0.3440 | 0.3636 | 0.3994 | 0.4788 | 7/1 |
| NV round 1 | 0.7000 | 0.7070 | 0.7538 | 0.7254 | 0.3670 | 0.3370 | 0.3939 | 0.3691 | 0.5442 | 5/3 |
| TS round 1 | 0.7850 | 0.7800 | 0.7003 | 0.6861 | 0.3420 | 0.3420 | 0.3967 | 0.3939 | 0.5533 | 6/2 |
| bestof4 round 1 | 0.7210 | 0.7350 | 0.7104 | 0.7003 | 0.3300 | 0.3390 | 0.3857 | 0.3802 | 0.5377 | 6/2 |
| DPO round 1 | 0.7190 | 0.6510 | 0.7012 | 0.6694 | 0.2850 | 0.2920 | 0.3554 | 0.3554 | 0.5035 | 7/1 |
| NV round 3 | 0.6210 | 0.5820 | 0.6953 | 0.6661 | 0.3100 | 0.3040 | 0.3774 | 0.3416 | 0.4872 | 8/0 |
| TS round 3 | 0.7750 | 0.7690 | 0.6845 | 0.6594 | 0.3250 | 0.3440 | 0.3829 | 0.3802 | 0.5400 | 6/2 |
| **Repair-TS round 1** | 0.6990 | 0.7260 | 0.7379 | 0.7070 | 0.3790 | 0.3550 | 0.4187 | 0.4215 | **0.5555** | |

Mean of 3 seeds per cell; per-seed means of 8; repair-TS cells higher/lower over 24 (seed, cell) pairs, ties not counted:

| Arm | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean | Seeds 1, 2, 3 | Repair-TS higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 0.3260 | 0.6850 | 0.7245 | **0.7688** | 0.2190 | 0.3440 | 0.3636 | 0.3994 | 0.4788 | 0.4788, 0.4788, 0.4788 | 16/7 |
| NV round 1 | 0.6920 | 0.6840 | **0.7535** | 0.7218 | **0.3570** | 0.3337 | **0.3994** | 0.3792 | 0.5401 | 0.5418, 0.5342, 0.5442 | 14/10 |
| TS round 1 | 0.7230 | 0.7207 | 0.7028 | 0.6950 | 0.3010 | 0.3217 | 0.3701 | 0.3783 | 0.5266 | 0.5106, 0.5159, 0.5533 | 20/4 |
| bestof4 round 1 | 0.7157 | 0.7260 | 0.7070 | 0.6984 | 0.3140 | 0.3330 | 0.3866 | 0.3829 | 0.5330 | 0.5171, 0.5441, 0.5377 | 17/5 |
| DPO round 1 | 0.7227 | 0.6957 | 0.6981 | 0.6783 | 0.2573 | 0.3010 | 0.3425 | 0.3499 | 0.5057 | 0.4859, 0.5276, 0.5035 | 22/2 |
| NV round 3 | 0.6627 | 0.6170 | 0.6800 | 0.6555 | 0.3160 | 0.3040 | 0.3719 | 0.3471 | 0.4943 | 0.5067, 0.4890, 0.4872 | 23/0 |
| TS round 3 | **0.7790** | **0.7683** | 0.7031 | 0.6881 | 0.3217 | 0.3300 | 0.3802 | 0.3774 | 0.5435 | 0.5660, 0.5244, 0.5400 | 14/10 |
| **Repair-TS round 1** | 0.7387 | 0.7400 | 0.7254 | 0.7078 | 0.3530 | **0.3443** | 0.3949 | **0.4031** | **0.5509** | 0.5388, 0.5584, 0.5555 | |

- Repair-TS round 1 has the highest 3-seed mean of every arm: +0.0243 over TS round 1, +0.0179 over bestof4, +0.0108 over NV round 1, +0.0074 over TS round 3, +0.0721 over vanilla.
- It is above TS round 1 on all 3 seeds (+0.0282, +0.0425, +0.0022). The seed 3 margin is small because TS round 1 seed 3 is its strongest seed (0.5533).
- It is above TS round 3 on seeds 2 and 3 and below on seed 1 (−0.0272).
- **Outcome: round 1 improves on TS.** Per the plan, rounds 2-3 are next.

### Extension 3, round 2, seed 2 (finished 2026-11-09, 16:40 IST)

Round 2 (`ts/repair_loop.py`): 2,916 questions; Algorithm 3 kept 713 (TS's own round 2 on the same tables, with TS's adapters, is a different run). Repair: 156 of 2,148 recovered (disagree 116 of 889, Scala failed 23 of 515, always empty 14 of 688, SQL failed 3 of 56); kept 713 → 869 (+21.9%). 4 pairs were rejected by an `OverflowError` in SQLite (an integer too large); TS's round 2 for seed 2 has the same 4.

Times: generation 31 min, validation 3 h 12 min, repair 9 h 38 min (includes waiting for the validation slot), training and evaluation 2 h 10 min.

| Arm, seed 2 | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 | Repair-TS r2 higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|
| TS round 1 | 0.7040 | 0.7330 | 0.6928 | 0.6953 | 0.2690 | 0.3110 | 0.3471 | 0.3747 | 0.5159 | 8/0 |
| TS round 2 | 0.7510 | 0.7490 | 0.6803 | 0.6811 | 0.2700 | 0.3000 | 0.3196 | 0.3361 | 0.5109 | 8/0 |
| TS round 3 | 0.7720 | 0.7640 | 0.6753 | 0.6711 | 0.2860 | 0.3080 | 0.3526 | 0.3664 | 0.5244 | 7/1 |
| NV round 2 | 0.6380 | 0.6290 | 0.7020 | 0.6669 | 0.3300 | 0.3040 | 0.3857 | 0.3691 | 0.5031 | 6/2 |
| Repair-TS round 1 | 0.7820 | 0.7590 | 0.7337 | 0.7220 | 0.3370 | 0.3430 | 0.3912 | 0.3994 | 0.5584 | 3/4 (1 tie) |
| **Repair-TS round 2** | 0.7890 | 0.7620 | 0.7346 | 0.7220 | 0.3290 | 0.3290 | 0.3829 | 0.3857 | **0.5543** | |

Repair-TS round 2 is +0.0434 over TS round 2 and +0.0299 over TS round 3, but −0.0041 under its own round 1: higher on WikiSQL and Spider z, lower on WikiTQ and BIRD.

### Extension 3, round 2, all 3 seeds (seed 1 finished 2026-11-09, 19:43 IST; seed 3 21:15 IST)

Repair in round 2: seed 1 140 of 2,134 recovered, kept 690 → 830 (+20.3%); seed 2 156 of 2,148, 713 → 869 (+21.9%); seed 3 178 of 2,106, 701 → 879 (+25.4%).

| Seed | Arm | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 | Repair-TS r2 higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TS round 2 | 0.7560 | 0.7500 | 0.7287 | 0.7237 | 0.2970 | 0.3250 | 0.3774 | 0.3609 | 0.5398 | 7/1 |
| 1 | TS round 3 | 0.7900 | 0.7720 | 0.7496 | 0.7337 | 0.3540 | 0.3380 | 0.4050 | 0.3857 | 0.5660 | 6/2 |
| 1 | Repair-TS round 1 | 0.7350 | 0.7350 | 0.7045 | 0.6945 | 0.3430 | 0.3350 | 0.3747 | 0.3884 | 0.5388 | 8/0 |
| 1 | **Repair-TS round 2** | 0.7660 | 0.7360 | 0.7613 | 0.7571 | 0.3740 | 0.3660 | 0.4105 | 0.4022 | **0.5716** | |
| 3 | TS round 2 | 0.7310 | 0.7240 | 0.7062 | 0.6953 | 0.3240 | 0.3310 | 0.3857 | 0.3829 | 0.5350 | 8/0 |
| 3 | TS round 3 | 0.7750 | 0.7690 | 0.6845 | 0.6594 | 0.3250 | 0.3440 | 0.3829 | 0.3802 | 0.5400 | 6/2 |
| 3 | Repair-TS round 1 | 0.6990 | 0.7260 | 0.7379 | 0.7070 | 0.3790 | 0.3550 | 0.4187 | 0.4215 | 0.5555 | 8/0 |
| 3 | **Repair-TS round 2** | 0.7640 | 0.7560 | 0.7938 | 0.7513 | 0.4070 | 0.3840 | 0.4242 | 0.4242 | **0.5881** | |

Mean of 3 seeds (seed 2 in the section above); per-seed means of 8:

| Arm | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean | Seeds 1, 2, 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 0.3260 | 0.6850 | 0.7245 | **0.7688** | 0.2190 | 0.3440 | 0.3636 | 0.3994 | 0.4788 | same |
| TS round 1 | 0.7230 | 0.7207 | 0.7028 | 0.6950 | 0.3010 | 0.3217 | 0.3701 | 0.3783 | 0.5266 | 0.5106, 0.5159, 0.5533 |
| TS round 2 | 0.7460 | 0.7410 | 0.7051 | 0.7000 | 0.2970 | 0.3187 | 0.3609 | 0.3600 | 0.5286 | 0.5398, 0.5109, 0.5350 |
| TS round 3 | **0.7790** | **0.7683** | 0.7031 | 0.6881 | 0.3217 | 0.3300 | 0.3802 | 0.3774 | 0.5435 | 0.5660, 0.5244, 0.5400 |
| NV round 1 | 0.6920 | 0.6840 | 0.7535 | 0.7218 | 0.3570 | 0.3337 | 0.3994 | 0.3792 | 0.5401 | 0.5418, 0.5342, 0.5442 |
| NV round 2 | 0.6600 | 0.6483 | 0.7167 | 0.6895 | 0.3397 | 0.3163 | 0.3939 | 0.3737 | 0.5173 | 0.5236, 0.5031, 0.5252 |
| bestof4 round 1 | 0.7157 | 0.7260 | 0.7070 | 0.6984 | 0.3140 | 0.3330 | 0.3866 | 0.3829 | 0.5330 | 0.5171, 0.5441, 0.5377 |
| Repair-TS round 1 | 0.7387 | 0.7400 | 0.7254 | 0.7078 | 0.3530 | 0.3443 | 0.3949 | 0.4031 | 0.5509 | 0.5388, 0.5584, 0.5555 |
| **Repair-TS round 2** | 0.7730 | 0.7513 | **0.7632** | 0.7435 | **0.3700** | **0.3597** | **0.4059** | **0.4040** | **0.5713** | 0.5716, 0.5543, 0.5881 |

- Repair-TS round 2 has the highest 3-seed mean of every arm: +0.0427 over TS round 2, +0.0278 over TS round 3, +0.0312 over NV round 1, +0.0204 over repair-TS round 1, +0.0925 over vanilla.
- Above TS round 2 on all 3 seeds (+0.0318, +0.0434, +0.0531) and above TS round 3 on all 3 (+0.0056, +0.0299, +0.0481).
- Above its own round 1 on seeds 1 and 3 (+0.0328, +0.0326), below on seed 2 (−0.0041).
- Best of all arms on 6 of 8 cells; TS round 3 leads WikiSQL, vanilla leads Spider few-shot.

### Extension 3, round 3, seed 2 (finished 2026-11-10, 11:18 IST, exit 0)

Repair: 151 of 2,010 recovered (disagree 106 of 833, Scala failed 33 of 568, always empty 10 of 556, SQL failed 2 of 53); kept 829 → 980 (+18.2%). Times: generation 33 min, validation 5 h 4 min (includes waiting for the slot), repair 10 h 39 min (includes waiting), training and evaluation 2 h 22 min.

| Arm, seed 2 | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 | Repair-TS r3 higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|
| TS round 3 | 0.7720 | 0.7640 | 0.6753 | 0.6711 | 0.2860 | 0.3080 | 0.3526 | 0.3664 | 0.5244 | 6/2 |
| NV round 3 | 0.6750 | 0.6420 | 0.6628 | 0.6361 | 0.3080 | 0.2830 | 0.3636 | 0.3416 | 0.4890 | 8/0 |
| Repair-TS round 1 | 0.7820 | 0.7590 | 0.7337 | 0.7220 | 0.3370 | 0.3430 | 0.3912 | 0.3994 | 0.5584 | 6/2 |
| Repair-TS round 2 | 0.7890 | 0.7620 | 0.7346 | 0.7220 | 0.3290 | 0.3290 | 0.3829 | 0.3857 | 0.5543 | 6/2 |
| **Repair-TS round 3** | 0.7690 | 0.7580 | 0.7604 | 0.7462 | 0.3930 | 0.3880 | 0.4242 | 0.4022 | **0.5801** | |

Repair-TS round 3, seed 2: +0.0557 over TS round 3, +0.0217 over its own round 1, +0.0258 over its round 2.

### Extension 3, round 3, seed 1 (finished 2026-11-10, 13:09 IST, exit 0)

Repair: 161 of 2,048 recovered (disagree 113 of 817, Scala failed 37 of 545, always empty 9 of 612, SQL failed 2 of 74); kept 761 → 922 (+21.2%).

| Arm, seed 1 | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 | Repair-TS r3 higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|
| TS round 3 | 0.7900 | 0.7720 | 0.7496 | 0.7337 | 0.3540 | 0.3380 | 0.4050 | 0.3857 | 0.5660 | 7/1 |
| NV round 3 | 0.6920 | 0.6270 | 0.6820 | 0.6644 | 0.3300 | 0.3250 | 0.3747 | 0.3581 | 0.5067 | 8/0 |
| Repair-TS round 1 | 0.7350 | 0.7350 | 0.7045 | 0.6945 | 0.3430 | 0.3350 | 0.3747 | 0.3884 | 0.5388 | 8/0 |
| Repair-TS round 2 | 0.7660 | 0.7360 | 0.7613 | 0.7571 | 0.3740 | 0.3660 | 0.4105 | 0.4022 | 0.5716 | 6/1 (1 tie) |
| **Repair-TS round 3** | 0.7600 | 0.7780 | 0.7963 | 0.7638 | 0.3920 | 0.3860 | 0.4132 | 0.4022 | **0.5864** | |

Seed 1 was TS round 3's strongest seed; repair-TS round 3 is +0.0204 over it, +0.0148 over its own round 2.

### Extension 3, round 3, seed 3, and final 3-seed results (seed 3 finished 2026-11-10, 14:39 IST, exit 0)

Seed 3 round 3 repair: 168 of 1,905 recovered (disagree 122 of 772, Scala failed 28 of 478, always empty 16 of 579, SQL failed 2 of 76); kept 896 → 1,064 (+18.8%).

| Arm, seed 3 | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean of 8 | Repair-TS r3 higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|
| TS round 3 | 0.7750 | 0.7690 | 0.6845 | 0.6594 | 0.3250 | 0.3440 | 0.3829 | 0.3802 | 0.5400 | 6/2 |
| NV round 3 | 0.6210 | 0.5820 | 0.6953 | 0.6661 | 0.3100 | 0.3040 | 0.3774 | 0.3416 | 0.4872 | 8/0 |
| Repair-TS round 1 | 0.6990 | 0.7260 | 0.7379 | 0.7070 | 0.3790 | 0.3550 | 0.4187 | 0.4215 | 0.5555 | 6/2 |
| Repair-TS round 2 | 0.7640 | 0.7560 | 0.7938 | 0.7513 | 0.4070 | 0.3840 | 0.4242 | 0.4242 | 0.5881 | 5/3 |
| **Repair-TS round 3** | 0.7650 | 0.7650 | 0.7963 | 0.7771 | 0.3990 | 0.3950 | 0.4160 | 0.4077 | **0.5901** | |

Repair per round, 3 seeds: round 1 645 recovered (+52.1% kept), round 2 474 (+22.5%), round 3 480 (+19.3%).

**Final: mean of 3 seeds per cell; per-seed means of 8; repair-TS round 3 higher/lower over the 24 (seed, cell) pairs**

| Arm | WikiSQL z | WikiSQL f | Spider z | Spider f | WikiTQ z | WikiTQ f | BIRD z | BIRD f | Mean | Seeds 1, 2, 3 | RTS r3 higher/lower |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 0.3260 | 0.6850 | 0.7245 | **0.7688** | 0.2190 | 0.3440 | 0.3636 | 0.3994 | 0.4788 | same | 22/2 |
| NV round 1 | 0.6920 | 0.6840 | 0.7535 | 0.7218 | 0.3570 | 0.3337 | 0.3994 | 0.3792 | 0.5401 | 0.5418, 0.5342, 0.5442 | 24/0 |
| NV round 2 | 0.6600 | 0.6483 | 0.7167 | 0.6895 | 0.3397 | 0.3163 | 0.3939 | 0.3737 | 0.5173 | 0.5236, 0.5031, 0.5252 | 24/0 |
| NV round 3 | 0.6627 | 0.6170 | 0.6800 | 0.6555 | 0.3160 | 0.3040 | 0.3719 | 0.3471 | 0.4943 | 0.5067, 0.4890, 0.4872 | 24/0 |
| TS round 1 | 0.7230 | 0.7207 | 0.7028 | 0.6950 | 0.3010 | 0.3217 | 0.3701 | 0.3783 | 0.5266 | 0.5106, 0.5159, 0.5533 | 22/2 |
| TS round 2 | 0.7460 | 0.7410 | 0.7051 | 0.7000 | 0.2970 | 0.3187 | 0.3609 | 0.3600 | 0.5286 | 0.5398, 0.5109, 0.5350 | 24/0 |
| TS round 3 | **0.7790** | **0.7683** | 0.7031 | 0.6881 | 0.3217 | 0.3300 | 0.3802 | 0.3774 | 0.5435 | 0.5660, 0.5244, 0.5400 | 19/5 |
| bestof4 round 1 | 0.7157 | 0.7260 | 0.7070 | 0.6984 | 0.3140 | 0.3330 | 0.3866 | 0.3829 | 0.5330 | 0.5171, 0.5441, 0.5377 | 23/1 |
| DPO round 1 | 0.7227 | 0.6957 | 0.6981 | 0.6783 | 0.2573 | 0.3010 | 0.3425 | 0.3499 | 0.5057 | 0.4859, 0.5276, 0.5035 | 23/1 |
| Repair-TS round 1 | 0.7387 | 0.7400 | 0.7254 | 0.7078 | 0.3530 | 0.3443 | 0.3949 | 0.4031 | 0.5509 | 0.5388, 0.5584, 0.5555 | 20/4 |
| Repair-TS round 2 | 0.7730 | 0.7513 | 0.7632 | 0.7435 | 0.3700 | 0.3597 | 0.4059 | **0.4040** | 0.5713 | 0.5716, 0.5543, 0.5881 | 17/6 |
| **Repair-TS round 3** | 0.7647 | 0.7670 | **0.7843** | 0.7624 | **0.3947** | **0.3897** | **0.4178** | **0.4040** | **0.5856** | 0.5864, 0.5801, 0.5901 | |

- **Repair-TS round 3 has the highest 3-seed mean of every arm: 0.5856.** +0.0421 over TS round 3 (0.5435), +0.0455 over NV round 1 (the best NV round), +0.0590 over TS round 1, +0.1068 over vanilla.
- Above TS round 3 on every seed (+0.0204, +0.0557, +0.0501) and on 19 of 24 (seed, cell) pairs.
- Repair-TS improves with rounds on the 3-seed mean (0.5509 → 0.5713 → 0.5856), while TS gains less (0.5266 → 0.5286 → 0.5435) and NV falls (0.5401 → 0.5173 → 0.4943).
- Best arm on 5 of 8 cells (BIRD few-shot tied with its own round 2). TS round 3 still leads both WikiSQL cells (by 0.0143 and 0.0013); vanilla still leads Spider few-shot (by 0.0064).
- Zero-shot against the paper's TS (Llama-3.1-8B): WikiSQL 0.7647 vs 0.816, Spider 0.7843 vs 0.690, WikiTQ 0.3947 vs 0.449, BIRD 0.4178 vs 0.438.
- Caveat for the write-up: the repairs come from a second model (Llama-3.1-8B-Instruct); questions and first-pass code still come from base Llama-3.1-8B and its adapters.

## Final results table, all approaches (2026-11-10)

Rule: an approach's final score is its **last round** (round 3 for TS, NV and repair-TS; round 1 for bestof4 and DPO, which were run for one round), as the **mean ± standard deviation over seeds 1-3** (sample SD, n = 3). Vanilla is greedy and has one score. The paper's prompt is not stated; its Table 3 (Llama3.1-8B) is set beside our zero-shot prompt, the repo's default. Text2Analysis was not run.

**Zero-shot prompt (compare with the paper)**

| Approach | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Paper: vanilla | 0.525 | 0.670 | 0.300 | 0.388 | 0.4708 |
| Paper: Table-Specialist | **0.816** | 0.690 | **0.449** | **0.438** | **0.5983** |
| Vanilla (ours) | 0.3260 | 0.7245 | 0.2190 | 0.3636 | 0.4083 |
| No-validation, round 3 | 0.6627 ± 0.0371 | 0.6800 ± 0.0163 | 0.3160 ± 0.0122 | 0.3719 ± 0.0073 | 0.5077 ± 0.0104 |
| Table-Specialist, round 3 | 0.7790 ± 0.0096 | 0.7031 ± 0.0405 | 0.3217 ± 0.0341 | 0.3802 ± 0.0263 | 0.5460 ± 0.0268 |
| bestof4, round 1 | 0.7157 ± 0.0462 | 0.7070 ± 0.0277 | 0.3140 ± 0.0155 | 0.3866 ± 0.0042 | 0.5308 ± 0.0184 |
| DPO, round 1 | 0.7227 ± 0.0396 | 0.6981 ± 0.0206 | 0.2573 ± 0.0266 | 0.3425 ± 0.0349 | 0.5051 ± 0.0275 |
| Repair-TS, round 3 | 0.7647 ± 0.0045 | **0.7843 ± 0.0207** | 0.3947 ± 0.0038 | 0.4178 ± 0.0057 | 0.5904 ± 0.0037 |

**Few-shot prompt (ours only)**

| Approach | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Vanilla | 0.6850 | **0.7688** | 0.3440 | 0.3994 | 0.5493 |
| No-validation, round 3 | 0.6170 ± 0.0312 | 0.6555 ± 0.0169 | 0.3040 ± 0.0210 | 0.3471 ± 0.0095 | 0.4809 ± 0.0111 |
| Table-Specialist, round 3 | **0.7683 ± 0.0040** | 0.6881 ± 0.0400 | 0.3300 ± 0.0193 | 0.3774 ± 0.0099 | 0.5410 ± 0.0152 |
| bestof4, round 1 | 0.7260 ± 0.0422 | 0.6984 ± 0.0090 | 0.3330 ± 0.0104 | 0.3829 ± 0.0047 | 0.5351 ± 0.0098 |
| DPO, round 1 | 0.6957 ± 0.0697 | 0.6783 ± 0.0184 | 0.3010 ± 0.0173 | 0.3499 ± 0.0055 | 0.5062 ± 0.0198 |
| Repair-TS, round 3 | 0.7670 ± 0.0101 | 0.7624 ± 0.0155 | **0.3897 ± 0.0047** | **0.4040 ± 0.0032** | **0.5808 ± 0.0065** |

**One number per approach (mean of all 8 cells)**

| Approach | Round 1 | Round 2 | Round 3 | Final (last round) | Mean of all rounds and seeds | Best round |
|---|---|---|---|---|---|---|
| Paper: vanilla (zero-shot, 4 sets) | | | | 0.4708 | | |
| Paper: Table-Specialist (zero-shot, 4 sets) | | | | 0.5983 | | not stated |
| Vanilla | | | | 0.4788 | | |
| No-validation | 0.5401 | 0.5173 | 0.4943 | 0.4943 ± 0.0108 | 0.5172 | 1 |
| Table-Specialist | 0.5266 | 0.5286 | 0.5435 | 0.5435 ± 0.0210 | 0.5329 | 3 |
| bestof4 | 0.5330 | on hold | on hold | 0.5330 ± 0.0141 | 0.5330 | 1 |
| DPO | 0.5057 | not run | not run | 0.5057 ± 0.0209 | 0.5057 | 1 |
| **Repair-TS** | 0.5509 | 0.5713 | 0.5856 | **0.5856 ± 0.0051** | **0.5693** | 3 |

- Final ranking (mean of 8): repair-TS 0.5856 > TS 0.5435 > bestof4 0.5330 > DPO 0.5057 > NV 0.4943 > vanilla 0.4788.
- Zero-shot against the paper: repair-TS 0.5904 against the paper's TS 0.5983 (−0.0079); our TS 0.5460 (−0.0523). Repair-TS is above the paper on Spider (+0.0943), below on WikiSQL (−0.0513), WikiTQ (−0.0543) and BIRD (−0.0202). Our vanilla is below the paper's on WikiSQL and WikiTQ (see the baseline gate), so the paper's gains start from a higher base on those two.
- Repair-TS has the smallest seed spread of every fine-tuned approach (SD 0.0051 on the mean of 8; TS 0.0210).

## Plan A1: Instruct vanilla through the repo's chat call (2026-11-10, 22:31 IST)

Llama-3.1-8B-Instruct, chat, the repo's system message, 256 tokens, greedy; same test cases as every other result.

| Test set | Instruct zero-shot | Paper vanilla | Difference | Instruct few-shot | Base zero-shot |
|---|---|---|---|---|---|
| WikiSQL | 0.518 | 0.525 | −0.007 | 0.798 | 0.326 |
| WikiTQ | 0.289 | 0.300 | −0.011 | 0.423 | 0.219 |
| BIRD | 0.4022 | 0.388 | +0.014 | 0.4187 | 0.3636 |
| Spider | 0.7913 | 0.670 | +0.121 | 0.7963 | 0.7245 |

- **Gate passed on WikiSQL, WikiTQ and BIRD (all within ±0.015); Spider misses by +0.121**, as it does for base (+0.055) and with every prompt. Spider's gap is therefore attributed to the case set or scoring, not the model.
- Zero-shot is the prompt that matches; few-shot overshoots WikiSQL by 0.27.
- **Outcome: go to A2** (TS on Instruct).

## Plan B1 dev tuning: self-consistency temperature (finished 22:31 IST)

TS round 3 seed 1 adapter, zero-shot, greedy + 8 samples. Dev sets: WikiSQL 300, Spider 411 (575 minus those over the prompt limit), WikiTQ 300.

| Setting | WikiSQL | Spider | WikiTQ | Mean | Votes wrong → right / right → wrong |
|---|---|---|---|---|---|
| Greedy | 0.7733 | 0.7908 | 0.3033 | 0.6225 | |
| T 0.5 | 0.8167 | 0.8029 | 0.3133 | 0.6443 | 23 / 2 |
| T 0.7 | 0.8233 | 0.8054 | 0.3167 | 0.6485 | 27 / 2 |
| **T 1.0** | **0.8300** | **0.8127** | 0.3133 | **0.6520** | 31 / 2 |

**Outcome (rule fixed beforehand): T 1.0, +0.0295 over greedy on dev, goes to the test sets.** Test runs started 22:32 IST in tmux `sc_test`, GPU 1: vanilla, TS round 3 and repair-TS round 3, seeds 1-3, 8 cells each.

## Plan B1 + B2 test sets: self-consistency at T 1.0 (finished 2026-11-11, 15:35 IST, exit 0)

Greedy + 8 samples at T 1.0; failed and empty results dropped; majority vote on results; the gold answer is never used for the choice. 56 cells (7 models × 8). Files `results/baseline/final_sct10_*.json`. Each "greedy" figure is the greedy answer from the same run, so each gain is measured within the run. Fine-tuned rows are mean ± SD over seeds 1-3.

**Zero-shot (compare with the paper)**

| Model | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Paper: Table-Specialist | 0.816 | 0.690 | 0.449 | 0.438 | 0.5983 |
| Vanilla, greedy → vote | 0.3260 → 0.3930 | 0.7245 → 0.7279 | 0.2190 → 0.2390 | 0.3554 → 0.3636 | 0.4062 → 0.4309 |
| TS r3, greedy → vote | 0.7790 → 0.8060 | 0.7031 → 0.7059 | 0.3217 → 0.3323 | 0.3719 → 0.3848 | 0.5439 ± 0.0268 → 0.5573 ± 0.0251 |
| Repair-TS r3, greedy → vote | 0.7647 → 0.8007 | 0.7843 → 0.7908 | 0.3947 → 0.4103 | 0.4086 → 0.4187 | 0.5881 ± 0.0038 → **0.6051 ± 0.0084** |

**Few-shot**

| Model | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Vanilla, greedy → vote | 0.6850 → 0.7680 | 0.7688 → 0.7721 | 0.3440 → 0.3600 | 0.3912 → 0.3884 | 0.5473 → 0.5721 |
| TS r3, greedy → vote | 0.7683 → 0.8057 | 0.6881 → 0.7003 | 0.3300 → 0.3443 | 0.3692 → 0.3848 | 0.5389 ± 0.0148 → 0.5588 ± 0.0150 |
| Repair-TS r3, greedy → vote | 0.7670 → 0.8070 | 0.7624 → 0.7685 | 0.3897 → 0.4040 | 0.3949 → 0.4004 | 0.5785 ± 0.0067 → 0.5950 ± 0.0079 |

**Mean of all 8 cells, greedy → vote:** vanilla 0.4767 → 0.5015 (+0.0248); TS r3 0.5414 → 0.5580 ± 0.0200 (+0.0166); repair-TS r3 0.5833 → **0.6000 ± 0.0079** (+0.0168).

- Self-consistency raises every model. Only 2 of 56 cells drop: vanilla BIRD few-shot −0.0028 and repair-TS seed 2 BIRD zero-shot −0.0027, each one case.
- Answers changed from wrong to right / right to wrong, summed over cells: vanilla 278 / 82, TS 462 / 109, repair-TS 514 / 134.
- The largest gains are on WikiSQL (+0.03 to +0.08). Spider gains are the smallest (+0.003 to +0.012).
- **Repair-TS + self-consistency, zero-shot, mean of 4: 0.6051, above the paper's TS at 0.5983 (+0.0068).** It is above the paper on Spider (+0.1008). It is below on WikiSQL (−0.0153), WikiTQ (−0.0387) and BIRD (−0.0193).
- **BIRD greedy is not the same as before.** The greedy BIRD score in this script is exactly 3 cases (0.0082) below the earlier `run_baseline` score, on every BIRD cell checked: vanilla zero-shot 0.3554 against 0.3636, and few-shot 0.3912 against 0.3994. The other test sets match.
  - **Likely cause, not yet checked:** slow BIRD queries timing out while 16 scorers run in parallel.
  - **Effect:** it lowers the greedy BIRD score and the vote BIRD score alike, so the BIRD gains still hold, but the absolute BIRD figures here may be about 0.008 low.
  - The means of 8 above use this script's greedy figures, so they sit about 0.002 below the "Final results" table.

### B1 BIRD re-scored on one thread (finished 2026-11-12, 17:57 IST, exit 0); corrected B1 totals

- **Cause, confirmed:** the 3 cases differ only in scoring. Each returns 20,198-49,729 rows and hit the 5 s SQL timeout while 16 scoring threads ran.
- **Fix:** BIRD is scored on one thread (`--score-workers 1`), in 14 new files `results/baseline/final_sct10seq_*_bird_*.json`. The old files are kept.
- **Check:** greedy BIRD now equals `run_baseline` exactly on every cell; for example vanilla 0.3636 / 0.3994, and TS seed 1 0.405 / 0.3857.

**BIRD, greedy → vote, 3-seed mean (zero-shot / few-shot):**
- Vanilla: 0.3636 → 0.3719 / 0.3994 → 0.3967.
- TS r3: 0.3802 → 0.3884 / 0.3774 → 0.3893.
- Repair-TS r3: 0.4178 → 0.4251 / 0.4040 → 0.4059.

**Corrected B1 totals (BIRD from the one-thread files; greedy now equals the "Final results" table):**

| Model | Zero-shot mean of 4, greedy → vote | Few-shot mean of 4, greedy → vote | Mean of 8, greedy → vote |
|---|---|---|---|
| Paper: Table-Specialist | 0.5983 | | |
| Vanilla | 0.4083 → 0.4330 | 0.5493 → 0.5742 | 0.4788 → 0.5036 (+0.0248) |
| TS r3 | 0.5460 → 0.5582 ± 0.0257 | 0.5410 → 0.5599 ± 0.0157 | 0.5435 → 0.5590 ± 0.0207 (+0.0156) |
| **Repair-TS r3** | 0.5904 → **0.6067 ± 0.0081** | 0.5808 → 0.5964 ± 0.0073 | 0.5856 → **0.6015 ± 0.0075** (+0.0160) |

These replace the B1 totals above. Base repair-TS + self-consistency, zero-shot mean of 4: **0.6067, +0.0084 above the paper's TS**.

## Plan A2 + A3: Instruct TS and Instruct NV, round 1 (as of 2026-11-11, 15:35 IST)

Llama-3.1-8B-Instruct; chat, with the repo system message, in generation, training and evaluation. Greedy, repo-faithful scoring. Run IDs `train_tablespecialist_instruct_seed{s}`, `train_novalidation_instruct_seed1`.

**Zero-shot (compare with the paper)**

| Model | Kept pairs | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|---|
| Paper: vanilla | | 0.525 | 0.670 | 0.300 | 0.388 | 0.4708 |
| Paper: Table-Specialist | | 0.816 | 0.690 | 0.449 | 0.438 | 0.5983 |
| Instruct vanilla (A1) | | 0.518 | 0.7913 | 0.289 | 0.4022 | 0.5001 |
| Instruct TS seed 1, round 1 | 319 of 3,000 | 0.801 | 0.818 | 0.422 | 0.4711 | **0.6280** |
| Instruct TS seed 2, round 1 | 332 of 3,000 | 0.808 | 0.798 | 0.411 | 0.4215 | **0.6096** |
| Instruct NV seed 1, round 1 | 2,881 (no check) | 0.781 | 0.8063 | 0.423 | 0.4545 | 0.6162 |

**Few-shot:**
- TS seed 1: WikiSQL 0.803, Spider 0.8155, WikiTQ 0.435, BIRD 0.4518.
- TS seed 2: WikiSQL 0.798, Spider 0.7788, WikiTQ 0.425, BIRD 0.416.
- NV seed 1: WikiSQL 0.759, Spider 0.7863, WikiTQ 0.424, BIRD 0.4077.

**Mean of 8 cells:** TS seed 1 0.6272, TS seed 2 0.6070, NV seed 1 0.6052.

- **Round 1 of Instruct TS is already above the paper's final TS** on the mean of 4 zero-shot sets (0.6280 and 0.6096 against 0.5983):
  - Close to the paper on WikiSQL (−0.008 to −0.015) and WikiTQ (−0.027 to −0.038).
  - Above it on BIRD for seed 1 (+0.033), level for seed 2 (−0.017).
  - Above it on Spider (+0.11 to +0.13), as with every model of ours.
  - **Vanilla → TS gain on WikiSQL:** +0.283 to +0.290 here, against the paper's +0.291. Base gave +0.453 from a lower start.
- **Pairs kept:** Instruct keeps 10.6-11.1% of tables, below base's 14.7%.
  - Seed 1 reasons: always_empty 1,007, scala_failed 691, disagree 626, sql_failed 234, no_code 97, SQLite integer overflow 4.
- **NV round 1 is between the two TS seeds on zero-shot** (0.6162), as base NV was at round 1. The paper's ablation is judged at round 3.
- **Seed 3** is still training round 1. Rounds 2-3 are running for all four jobs.

### A2/A3 progress (2026-11-12, 05:30 IST): Instruct TS seed 1 finished, exit 0

Zero-shot, then few-shot:

| Run, round | Kept | WikiSQL | Spider | WikiTQ | BIRD | Zero-shot mean of 4 | Mean of 8 |
|---|---|---|---|---|---|---|---|
| Paper: Table-Specialist | | 0.816 | 0.690 | 0.449 | 0.438 | 0.5983 | |
| TS seed 1, r1 | 319 | 0.801 / 0.803 | 0.818 / 0.8155 | 0.422 / 0.435 | 0.4711 / 0.4518 | 0.6280 | 0.6272 |
| TS seed 1, r2 | 499 | 0.814 / 0.796 | 0.8222 / 0.8114 | 0.445 / 0.454 | 0.4573 / 0.4435 | **0.6346** | **0.6304** |
| TS seed 1, r3 | 522 | 0.779 / 0.753 | 0.8139 / 0.8047 | 0.426 / 0.436 | 0.4435 / 0.438 | 0.6156 | 0.6118 |
| TS seed 2, r2 | 486 | 0.794 / 0.786 | 0.8022 / 0.783 | 0.422 / 0.427 | 0.449 / 0.4298 | 0.6168 | 0.6116 |
| TS seed 3, r1 | 332 | 0.764 / 0.772 | 0.813 / 0.8013 | 0.402 / 0.421 | 0.4683 / 0.449 | 0.6118 | 0.6113 |
| TS seed 3, r2 | 486 | 0.764 / 0.753 | 0.7988 / 0.7796 | 0.413 / 0.422 | 0.4298 / 0.4077 | 0.6014 | 0.5960 |
| NV seed 1, r2 | 2,899 | 0.786 / 0.754 | 0.7922 / 0.7654 | 0.427 / 0.407 | 0.4325 / 0.3994 | 0.6094 | 0.5954 |

- **Seed 1 peaks at round 2 and falls at round 3** (−0.019 on the zero-shot mean of 4). The fall comes mostly from WikiSQL (0.814 → 0.779) and BIRD.
  - Base TS rose at round 3 instead.
  - The paper does not say which round Table 3 reports (plan item A7). Both round 3 and the best round will be reported.
- **Seed 1 round 2 is the closest match to the paper so far:**
  - WikiSQL 0.814 against 0.816; WikiTQ 0.445 against 0.449.
  - BIRD 0.4573 against 0.438.
  - Spider 0.8222 against 0.690, above it as with every model of ours.
- **TS against NV at round 2:** TS seed 1 0.6346 and TS seed 2 0.6168, against NV seed 1 0.6094. TS seed 3 is lower (0.6014).

### A2 progress (2026-11-12, 11:00 IST): Instruct TS seed 2 finished, exit 0

| Run, round | Kept | WikiSQL | Spider | WikiTQ | BIRD | Zero-shot mean of 4 | Mean of 8 |
|---|---|---|---|---|---|---|---|
| TS seed 2, r3 | 529 | 0.756 / 0.763 | 0.7988 / 0.7713 | 0.422 / 0.426 | 0.4077 / 0.3912 | 0.5961 | 0.5920 |

Zero-shot mean of 4 by round, so far:

| Seed | r1 | r2 | r3 |
|---|---|---|---|
| 1 | 0.6280 | **0.6346** | 0.6156 |
| 2 | 0.6096 | **0.6168** | 0.5961 |
| 3 | 0.6118 | 0.6014 | running |

- **Both finished seeds peak at round 2 and fall at round 3:** −0.019 for seed 1 and −0.021 for seed 2.
  - On WikiSQL, both seeds fall about 0.04 at round 3 (0.814 → 0.779 and 0.794 → 0.756).
  - Base TS rose at round 3 instead (mean of 8: 0.5266, 0.5286, 0.5435).
- **Seed 2, round 3 against the paper's TS (0.5983):** 0.5961, about level.

### A3 finished: Instruct NV seed 1 (2026-11-12, 11:02 IST, exit 0)

| Run, round | Kept | WikiSQL | Spider | WikiTQ | BIRD | Zero-shot mean of 4 | Mean of 8 |
|---|---|---|---|---|---|---|---|
| NV seed 1, r1 | 2,881 | 0.781 / 0.759 | 0.8063 / 0.7863 | 0.423 / 0.424 | 0.4545 / 0.4077 | 0.6162 | 0.6052 |
| NV seed 1, r2 | 2,899 | 0.786 / 0.754 | 0.7922 / 0.7654 | 0.427 / 0.407 | 0.4325 / 0.3994 | 0.6094 | 0.5954 |
| NV seed 1, r3 | 2,861 | 0.726 / 0.693 | 0.7805 / 0.7487 | 0.413 / 0.392 | 0.4215 / 0.3967 | 0.5852 | 0.5714 |

- **Decision rule (fixed beforehand), TS must beat NV at round 3: passed for both finished TS seeds.**

  | Round 3 | Zero-shot mean of 4 | Mean of 8 |
  |---|---|---|
  | TS seed 1 | 0.6156 | 0.6118 |
  | TS seed 2 | 0.5961 | 0.5920 |
  | NV seed 1 | 0.5852 | 0.5714 |

- **NV falls every round** (0.6162 → 0.6094 → 0.5852), as base NV did (mean of 8: 0.5401 → 0.5173 → 0.4943). This is the paper's ablation finding: without validation, later rounds get worse.
- **NV is only seed 1.** The TS-NV gap at round 3 (+0.011 to +0.030 on the zero-shot mean) is about the size of the TS seed spread. It is not tested across seeds.

### A2 finished: Instruct TS, 3 seeds × 3 rounds (seed 3 finished 2026-11-12, 12:54 IST, exit 0)

Seed 3, round 3:
- 526 pairs kept.
- Zero-shot / few-shot: WikiSQL 0.818 / 0.789, Spider 0.8047 / 0.7938, WikiTQ 0.422 / 0.423, BIRD 0.4435 / 0.4408.
- Zero-shot mean of 4: 0.6220. Mean of 8: 0.6169.

**Zero-shot, mean ± SD over seeds 1-3 (compare with the paper)**

| | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Paper: vanilla | 0.525 | 0.670 | 0.300 | 0.388 | 0.4708 |
| Paper: Table-Specialist | 0.816 | 0.690 | 0.449 | 0.438 | 0.5983 |
| Instruct vanilla (A1) | 0.518 | 0.7913 | 0.289 | 0.4022 | 0.5001 |
| Instruct TS, r1 | 0.7910 ± 0.0236 | 0.8097 ± 0.0104 | 0.4117 ± 0.0100 | 0.4536 ± 0.0279 | 0.6165 ± 0.0100 |
| Instruct TS, r2 | 0.7907 ± 0.0252 | 0.8077 ± 0.0126 | 0.4267 ± 0.0165 | 0.4454 ± 0.0141 | 0.6176 ± 0.0166 |
| **Instruct TS, r3** | **0.7843 ± 0.0313** | 0.8058 ± 0.0076 | **0.4233 ± 0.0023** | **0.4316 ± 0.0207** | **0.6113 ± 0.0135** |
| Base TS, r3 (faithful reproduction) | 0.7790 | 0.7031 | 0.3217 | 0.3802 | 0.5460 |

**Few-shot, mean ± SD over seeds 1-3**

| | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Instruct TS, r1 | 0.7910 ± 0.0166 | 0.7985 ± 0.0185 | 0.4270 ± 0.0072 | 0.4389 ± 0.0199 | 0.6139 ± 0.0113 |
| Instruct TS, r2 | 0.7783 ± 0.0225 | 0.7913 ± 0.0175 | 0.4343 ± 0.0172 | 0.4270 ± 0.0181 | 0.6078 ± 0.0179 |
| Instruct TS, r3 | 0.7683 ± 0.0186 | 0.7899 ± 0.0170 | 0.4283 ± 0.0068 | 0.4233 ± 0.0279 | 0.6025 ± 0.0128 |

**Mean of 8 cells:**
- By round: r1 0.6152 ± 0.0106, r2 0.6127 ± 0.0172, r3 0.6069 ± 0.0131.
- Mean over all rounds and seeds: 0.6116.
- Best round per seed (zero-shot mean of 4): 0.6346, 0.6168, 0.6220; mean 0.6245.

**Against the paper (Instruct TS, round 3, zero-shot)**

| Test set | Ours − paper | Our gain over vanilla | Paper's gain over vanilla |
|---|---|---|---|
| WikiSQL | −0.032 | +0.266 | +0.291 |
| WikiTQ | −0.026 | +0.134 | +0.149 |
| BIRD | −0.006 | +0.029 | +0.050 |
| Spider | +0.116 | +0.015 | +0.020 |
| Mean of 4 | **+0.013** | | |

- **WikiSQL, WikiTQ and BIRD are all within 0.032 of the paper**, and well within the ±0.05 gate.
  - On these three sets the mean is 0.5464, against the paper's 0.5677 (−0.021).
  - Spider is above the paper with every model of ours, vanilla included.
- **The gains over vanilla match the paper in pattern and come close in size on every test set.**
  - Both are largest on WikiSQL, then WikiTQ, and smallest on BIRD and Spider.
  - Ours are 0.005-0.025 smaller.
- **Conclusion for Goal A:** Instruct TS reproduces the paper's Llama Table 3 to within about 0.03 per test set (Spider excepted).
  - This supports, but does not prove, the reading that the paper used Llama-3.1-8B-Instruct through the repo's chat call.
  - Base TS stays the reproduction that follows the README's base link.
- **Rounds:** on the 3-seed mean, rounds 1-3 are flat (0.6165, 0.6176, 0.6113; differences below 1 SD).
  - Seeds 1-2 peak at round 2; seed 3 rises at round 3.
  - The paper says rounds help. On Instruct we see no gain after round 1, while base TS gained +0.017 from round 1 to round 3.
- **TS against NV (seed 1 only):** round 3, zero-shot mean of 4, TS 0.6113 ± 0.0135 against NV 0.5852.

## Plan step 4: B1 self-consistency on Instruct (finished 2026-11-12, 19:58 IST, exit 0)

- **Setup:** Instruct vanilla and Instruct TS round 3, seeds 1-3. Greedy + 8 samples at T 1.0, the T chosen on dev for base. Chat with the repo system message, 256 tokens.
- **Scope:** 32 cells; BIRD scored on one thread. Files `results/baseline/final_sct10_instruct_*.json`.
- **Check:** greedy equals the A2 loop's own evaluation, 0.6113 and 0.6069.

**Zero-shot (compare with the paper)**

| Model | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Paper: Table-Specialist | 0.816 | 0.690 | 0.449 | 0.438 | 0.5983 |
| Instruct vanilla, greedy → vote | 0.5180 → 0.5510 | 0.7913 → 0.7922 | 0.2890 → 0.2950 | 0.4022 → 0.4022 | 0.5001 → 0.5101 |
| Instruct TS r3, greedy → vote | 0.7843 → **0.8297** | 0.8058 → 0.8080 | 0.4233 → 0.4407 | 0.4316 → 0.4325 | 0.6113 ± 0.0135 → **0.6277 ± 0.0097** |

**Few-shot**

| Model | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Instruct vanilla, greedy → vote | 0.7980 → 0.8390 | 0.7963 → 0.8130 | 0.4230 → 0.4520 | 0.4187 → 0.4325 | 0.6090 → 0.6341 |
| Instruct TS r3, greedy → vote | 0.7683 → 0.8160 | 0.7899 → 0.7971 | 0.4283 → 0.4390 | 0.4233 → 0.4307 | 0.6025 ± 0.0128 → 0.6207 ± 0.0075 |

**Mean of 8, greedy → vote:**
- Instruct vanilla: 0.5546 → 0.5721 (+0.0176).
- Instruct TS r3: 0.6069 → 0.6242 ± 0.0086 (+0.0173). By seed: 0.6282, 0.6143, 0.6300.

- **Instruct TS + vote, zero-shot, is above the paper's TS on the mean of 4** (0.6277 against 0.5983).
  - WikiSQL +0.014.
  - Within 0.01 below the paper on WikiTQ (−0.008) and BIRD (−0.006).
- **Wrong → right / right → wrong:** vanilla 171 / 36; TS 504 / 98. Only 2 of 32 cells fall.
- **Instruct vanilla with the few-shot prompt is as strong as Instruct TS.**
  - Mean of 4, greedy: 0.6090 against 0.6025 (few-shot) and 0.6113 (zero-shot).
  - With the vote, few-shot vanilla (0.6341) is above zero-shot TS (0.6277).
  - TS's gain is a zero-shot gain, as in the paper's set-up: the paper does not state its test prompt, and the repo's default is zero-shot.
  - On base the picture differs: few-shot vanilla 0.5493, zero-shot TS r3 0.5460.
  - **This is a finding to report.**

## Plan step 4: repair-TS on Instruct, round 1 (as of 2026-11-13, 09:10 IST)

Each seed repairs the pairs its own Instruct TS round 1 rejected. The Instruct model is both the generator and the repairer.

| Seed | TS kept | Recovered (increase) | WikiSQL | Spider | WikiTQ | BIRD | Zero-shot mean of 4 | Mean of 8 |
|---|---|---|---|---|---|---|---|---|
| 1 | 319 | 74 (+23.2%) | 0.842 / 0.823 | 0.7955 / 0.7663 | 0.429 / 0.433 | 0.4435 / 0.4215 | 0.6275 | 0.6192 |
| 2 | 332 | 75 (+22.6%) | 0.839 / 0.811 | 0.8063 / 0.783 | 0.423 / 0.438 | 0.4325 / 0.4298 | 0.6252 | 0.6203 |
| 3 | 332 | 65 (+19.6%) | 0.694 / 0.708 | 0.8222 / 0.7947 | 0.400 / 0.436 | 0.4601 / 0.449 | 0.5941 | 0.5955 |

Cells are zero-shot / few-shot.

- **Against Instruct TS round 1, 3-seed means:**

  | Round 1, 3-seed mean | Zero-shot mean of 4 | Mean of 8 |
  |---|---|---|
  | Repair-TS | 0.6156 | 0.6117 |
  | TS | 0.6165 | 0.6152 |

  No gain at round 1 on Instruct; base gained +0.024 on the mean of 8.
- **Seeds differ:** seeds 1 and 2 gain on WikiSQL (+0.04 and +0.03 zero-shot) and on the mean of 4 (−0.0005 and +0.0156). Seed 3 falls on WikiSQL (0.764 → 0.694).
- **Recovery is lower than on base** (+19.6% to +23.2%, against base round 1's +52.1%).
  - Most recoveries are `disagree` pairs (seed 1: 51 of 626).
  - `always_empty` pairs, the largest rejected group on Instruct, are almost never recovered (5 of about 1,000).

## Significance tests (2026-11-13, 09:43 IST)

`python -m ts.analysis.significance`, file `results/analysis/significance.json`. Saved predictions only; no GPU.

- **Pairing:** each comparison is paired by test case. Per case, correctness is averaged over the seeds both models have (repo-faithful verdicts).
- **Statistic:** the mean over the 8 cells of A − B, each cell weighted equally. The zero-shot mean of 4 is also given.
- **Interval and p-value:** 95% interval and two-sided p from a paired bootstrap over test cases, 10,000 draws. "p 0.0000" means p < 0.0001.
- **Seed variation:** the case bootstrap does not cover it, so the per-seed differences are listed too.
- **Vanilla:** the greedy answers from the self-consistency runs, which equal the vanilla baselines exactly.

| Comparison | Seeds | Mean of 8: diff [95% CI], p | Zero-shot mean of 4: diff [95% CI], p | Per seed (mean of 8) |
|---|---|---|---|---|
| Base: TS r3 vs vanilla | 3 | +0.0647 [+0.0554, +0.0738], p 0.0000 | +0.1377 [+0.1228, +0.1527], p 0.0000 | +0.0872, +0.0456, +0.0612 |
| Base: TS r3 vs NV r3 | 3 | +0.0492 [+0.0426, +0.0557], p 0.0000 | +0.0384 [+0.0294, +0.0473], p 0.0000 | +0.0593, +0.0354, +0.0528 |
| Base: repair-TS r3 vs TS r3 | 3 | +0.0421 [+0.0367, +0.0476], p 0.0000 | +0.0444 [+0.0365, +0.0524], p 0.0000 | +0.0204, +0.0557, +0.0501 |
| Base: repair-TS r1 vs TS r1 | 3 | +0.0243 [+0.0197, +0.0290], p 0.0000 | +0.0288 [+0.0222, +0.0354], p 0.0000 | +0.0282, +0.0426, +0.0023 |
| Base: repair-TS r1 vs bestof4 r1 | 3 | +0.0180 [+0.0134, +0.0227], p 0.0000 | +0.0222 [+0.0157, +0.0288], p 0.0000 | +0.0217, +0.0144, +0.0178 |
| Base: DPO r1 vs TS r1 | 3 | −0.0209 [−0.0262, −0.0156], p 0.0000 | −0.0191 [−0.0263, −0.0118], p 0.0000 | −0.0246, +0.0117, −0.0497 |
| Instruct: TS r3 vs vanilla | 3 | +0.0523 [+0.0424, +0.0622], p 0.0000 | +0.1111 [+0.0956, +0.1264], p 0.0000 | +0.0572, +0.0374, +0.0623 |
| Instruct: TS r3 vs NV r3 | 1 | +0.0403 [+0.0308, +0.0499], p 0.0000 | +0.0304 [+0.0175, +0.0436], p 0.0000 | +0.0403 |
| Instruct: TS r3 vs TS r1 | 3 | −0.0083 [−0.0133, −0.0032], p 0.0012 | −0.0052 [−0.0124, +0.0019], p 0.1526 | −0.0154, −0.0150, +0.0055 |
| Instruct: repair-TS r1 vs TS r1 | 3 | −0.0035 [−0.0081, +0.0010], p 0.1268 | −0.0009 [−0.0072, +0.0053], p 0.7804 | −0.0080, +0.0133, −0.0158 |
| Vote vs greedy: base vanilla | 1 | +0.0248 [+0.0192, +0.0302], p 0.0000 | +0.0247 [+0.0196, +0.0299], p 0.0000 | |
| Vote vs greedy: base TS r3 | 3 | +0.0156 [+0.0132, +0.0180], p 0.0000 | +0.0122 [+0.0090, +0.0154], p 0.0000 | +0.0145, +0.0148, +0.0174 |
| Vote vs greedy: base repair-TS r3 | 3 | +0.0159 [+0.0132, +0.0187], p 0.0000 | +0.0163 [+0.0126, +0.0202], p 0.0000 | +0.0179, +0.0130, +0.0170 |
| Vote vs greedy: Instruct vanilla | 1 | +0.0175 [+0.0131, +0.0220], p 0.0000 | +0.0100 [+0.0070, +0.0132], p 0.0000 | |
| Vote vs greedy: Instruct TS r3 | 3 | +0.0173 [+0.0147, +0.0200], p 0.0000 | +0.0164 [+0.0128, +0.0203], p 0.0000 | +0.0165, +0.0224, +0.0132 |

- **Significant (p < 0.0001), with every seed in the same direction:**
  - TS over vanilla and over NV, on base and on Instruct.
  - Repair-TS over TS on base, at round 1 and round 3.
  - Repair-TS over bestof4 on base.
  - The vote over greedy for every model.
  - DPO below TS, though one seed of three is above.
- **Not significant:**
  - Instruct repair-TS r1 vs Instruct TS r1 (p 0.13 on the mean of 8; p 0.78 zero-shot).
  - Instruct TS r3 vs r1 on the zero-shot mean (p 0.15). On the mean of 8, round 3 is below round 1 (−0.0083, p 0.0012).
- **Pending:** the same-data control (base and Instruct) and Instruct repair-TS r3. The script lists them and reports them once their predictions exist.

## Same-data control for repair-TS (finished 2026-11-13, 22:00 IST, exit 0: base and Instruct, 3 seeds each)

`ts/data_control.py`: TS round 1's kept pairs, plus N kept pairs from another seed's TS round 1 on tables this seed never used. N is repair-TS round 1's recovered count, so the training-set sizes match repair-TS exactly. Run IDs `train_tsdata_{base,instruct}_seed{s}`. Cells are zero-shot / few-shot.

| Base, round 1 | Pairs | WikiSQL | Spider | WikiTQ | BIRD | Zero-shot mean of 4 | Mean of 8 |
|---|---|---|---|---|---|---|---|
| Control seed 1 | 426 + 224 | 0.742 / 0.731 | 0.7104 / 0.7053 | 0.304 / 0.317 | 0.3636 / 0.3829 | 0.5300 | 0.5320 |
| Control seed 2 | 388 + 217 | 0.736 / 0.752 | 0.727 / 0.722 | 0.339 / 0.341 | 0.3664 / 0.3802 | 0.5421 | 0.5454 |
| Control seed 3 | 423 + 204 | 0.737 / 0.745 | 0.7062 / 0.692 | 0.324 / 0.339 | 0.3747 / 0.3857 | 0.5355 | 0.5380 |
| **Control, 3-seed mean** | | | | | | **0.5359** | **0.5385** |
| TS r1, 3-seed mean | | | | | | 0.5242 | 0.5266 |
| Repair-TS r1, 3-seed mean | | | | | | 0.5530 | 0.5509 |

Paired bootstrap (as in "Significance tests"):

| Comparison | Mean of 8: diff [95% CI], p | Zero-shot mean of 4: diff [95% CI], p | Per seed (mean of 8) |
|---|---|---|---|
| Base: control r1 vs TS r1 | +0.0119 [+0.0075, +0.0163], p 0.0000 | +0.0116 [+0.0054, +0.0179], p 0.0000 | +0.0214, +0.0296, −0.0153 |
| Base: repair-TS r1 vs control r1 | +0.0124 [+0.0079, +0.0170], p 0.0000 | +0.0171 [+0.0106, +0.0237], p 0.0000 | +0.0067, +0.0130, +0.0176 |
| Instruct: control r1 vs TS r1 | +0.0079 [+0.0038, +0.0120], p 0.0006 | +0.0104 [+0.0044, +0.0164], p 0.0004 | +0.0068, +0.0096, +0.0072 |
| Instruct: repair-TS r1 vs control r1 | −0.0114 [−0.0155, −0.0074], p 0.0000 | −0.0113 [−0.0168, −0.0057], p 0.0000 | −0.0148, +0.0037, −0.0231 |

- **On base, repair-TS's round 1 gain over TS (+0.0243) splits about in half:**
  - **+0.0119 from the extra amount of data.** Not on every seed: seed 3 is −0.0153.
  - **+0.0124 from the repaired pairs themselves**, above the control on every seed.
  - So repaired pairs are worth more than the same number of extra first-pass pairs.
- **Instruct (3 seeds): the opposite of base.**

| Instruct, round 1 | Pairs | WikiSQL | Spider | WikiTQ | BIRD | Zero-shot mean of 4 | Mean of 8 |
|---|---|---|---|---|---|---|---|
| Control seed 1 | 319 + 74 | 0.821 / 0.815 | 0.8114 / 0.8005 | 0.430 / 0.430 | 0.4904 / 0.4738 | 0.6382 | 0.6340 |
| Control seed 2 | 332 + 75 | 0.828 / 0.817 | 0.8072 / 0.7746 | 0.403 / 0.427 | 0.4353 / 0.4408 | 0.6184 | 0.6166 |
| Control seed 3 | 332 + 65 | 0.786 / 0.770 | 0.8172 / 0.8063 | 0.414 / 0.435 | 0.4793 / 0.4408 | 0.6241 | 0.6186 |
| **Control, 3-seed mean** | | | | | | **0.6269** | **0.6231** |
| TS r1, 3-seed mean | | | | | | 0.6165 | 0.6152 |
| Repair-TS r1, 3-seed mean | | | | | | 0.6156 | 0.6117 |

  - Extra first-pass pairs help: control above TS r1 by +0.0079 (p 0.0006), on every seed.
  - Repaired pairs do not: repair-TS r1 is below the control by −0.0114 (p < 0.0001; 2 of 3 seeds).
  - Reading: on Instruct the repaired pairs are worth less than fresh first-pass pairs (recovery is lower, +20-23% against +52% on base, and concentrated in easier failure types), so repair-TS's value on Instruct is at most its data volume.

### Instruct repair-TS, round 2, all 3 seeds (as of 2026-11-13, 22:04 IST; round 3 running)

| Instruct, round 2 | Zero-shot mean of 4 | Mean of 8 |
|---|---|---|
| Repair-TS r2, 3-seed mean | 0.6241 ± 0.0047 | 0.6180 ± 0.0026 |
| TS r2, 3-seed mean | 0.6176 ± 0.0166 | 0.6127 |

- Repair-TS r2 vs TS r2: mean of 8 +0.0053 [+0.0008, +0.0099], p 0.021; zero-shot mean of 4 +0.0065, p 0.044. Per seed −0.0149, +0.0062, +0.0248, so not consistent across seeds.
- Seed 3 round 2: 479 kept + 92 recovered; zero-shot mean of 4 0.6268, mean of 8 0.6208.

#### Seed 1, round 3 (finished 2026-11-14, 05:53 IST, exit 0): all 3 rounds done

- Round 3 data: 2,974 pairs, 507 kept first pass, 89 recovered (+17.6%). By reason: disagree 51 of 654, scala_failed 29 of 521, always_empty 9 of 1,005, sql_failed 0 of 212.
- Test (zero-shot / few-shot): WikiSQL 0.826 / 0.808, Spider 0.7654 / 0.7396, WikiTQ 0.419 / 0.419, BIRD 0.4435 / 0.438.
- Zero-shot mean of 4 0.6135, mean of 8 0.6073. Instruct TS seed 1 round 3: 0.6156 / 0.6118. So −0.0045 on seed 1: repair-TS still does not beat TS on Instruct.
- Time: generation 0.7 h, validation 3.4 h, repair 7.4 h, training and scoring 2.4 h.

#### B1 vote on Instruct repair-TS r3, all 3 seeds (seed 2 finished 2026-11-14, 15:20 IST, exit 0): final

- Seed 2 vote (zero-shot / few-shot): WikiSQL 0.823 / 0.825, Spider 0.7947 / 0.783, WikiTQ 0.443 / 0.439, BIRD 0.4545 / 0.4132. Zero-shot mean of 4 0.6288, mean of 8 0.6219. Greedy matches all 8 training-run cells (as for seeds 1 and 3).
- **3-seed Instruct repair-TS r3 + vote** (zero-shot): WikiSQL 0.8370, WikiTQ 0.4427, BIRD 0.4573, Spider 0.7833; mean of 4 **0.6300 ± 0.0071**, mean of 8 0.6242 ± 0.0044.
- Vote vs greedy: +0.0188 [+0.0158, +0.0217], p < 0.0001.
- Against Instruct TS r3 + vote (mean of 8 0.6242 ± 0.0086): −0.0000 [−0.0045, +0.0046], p 0.997. Identical on average; lower seed-to-seed spread (SD 0.0044 against 0.0086).

#### Instruct repair-TS finished: all 3 seeds × 3 rounds (seed 2 finished 2026-11-14, 13:00 IST, exit 0)

- Seed 2 round 3: 525 kept first pass, 96 recovered (+18.3%). Test (zero-shot / few-shot): WikiSQL 0.758 / 0.774, Spider 0.7913 / 0.7755, WikiTQ 0.433 / 0.422, BIRD 0.4215 / 0.4022. Zero-shot mean of 4 0.6009, mean of 8 0.5972 (TS seed 2 r3: 0.5961 / 0.5920).
- **3-seed round 3** (zero-shot): WikiSQL 0.7920, WikiTQ 0.4327, BIRD 0.4444, Spider 0.7799; mean of 4 **0.6122 ± 0.0107** (TS r3 0.6113; paper TS 0.5983).
- Repair-TS r3 vs TS r3: mean of 8 −0.0014 [−0.0062, +0.0033], p 0.55; per seed −0.0044, +0.0052, −0.0050. **No difference on Instruct**, against +0.0421 (p < 0.0001) on base.

#### Seed 3, round 3 (finished 2026-11-14, 11:43 IST, exit 0)

- Round 3 data: 531 kept first pass, 81 recovered (+15.3%). By reason: disagree 52 of 621, scala_failed 22 of 532, always_empty 4 of 997, sql_failed 3 of 196.
- Test (zero-shot / few-shot): WikiSQL 0.792 / 0.777, Spider 0.783 / 0.7588, WikiTQ 0.446 / 0.437, BIRD 0.4683 / 0.4325.
- Zero-shot mean of 4 0.6223, mean of 8 0.6118. Instruct TS seed 3 round 3: 0.6220 / 0.6169, so −0.0050.
- Seeds 1 and 3 together, repair-TS r3 vs TS r3: mean of 8 −0.0047 [−0.0105, +0.0011], p 0.12.

#### B1 vote on Instruct repair-TS r3, seed 3 (finished 2026-11-14, 14:44 IST, exit 0)

- Greedy matches the training run's scoring in all 8 cells.
- Vote (zero-shot / few-shot): WikiSQL 0.837 / 0.829, Spider 0.7821 / 0.7646, WikiTQ 0.458 / 0.446, BIRD 0.4738 / 0.4435. Zero-shot mean of 4 0.6377, mean of 8 0.6292.

#### B1 vote on Instruct repair-TS r3, seed 1 (finished 2026-11-14, 09:15 IST, exit 0)

- Greedy matches the training run's scoring in all 8 cells.
- Vote (zero-shot / few-shot): WikiSQL 0.851 / 0.845, Spider 0.773 / 0.7479, WikiTQ 0.427 / 0.435, BIRD 0.4435 / 0.449. Zero-shot mean of 4 0.6236, mean of 8 0.6214.
- Vote vs greedy: +0.0141 (p < 0.0001). Against Instruct TS r3 + vote, seed 1 (0.6282): −0.0068, p 0.13.

#### Seed 1, round 2 (finished 2026-11-13)

- 500 pairs kept + 95 recovered.
- WikiSQL 0.814 / 0.800, Spider 0.793 / 0.7571, WikiTQ 0.424 / 0.416, BIRD 0.4766 / 0.4435.
- Zero-shot mean of 4: 0.6269; mean of 8: 0.6155.
- Instruct TS seed 1, round 2: 0.6346 / 0.6304.

## Other tasks in the paper (analysis, 2026-11-14)

> **Correction, 2026-11-18 (published paper read in full).** The EMNLP 2025 version (pp. 35443–35460), Table 3, confirms every Llama figure below; they are no longer unverified. It also gives NL-to-R per test set, which the summary had not: vanilla → TS WikiSQL 0.331 → 0.409, WikiTQ 0.138 → 0.257, Text2Analysis 0.103 → 0.199, Spider 0.503 → 0.536, BIRD 0.225 → 0.261 (mean of the 4 public sets 0.2993 → 0.3658). NL-to-Scala Text2Analysis: 0.129 → 0.214. The paper's averages (e.g. Scala 0.343, R 0.333) include Text2Analysis, so they are not inconsistent with the per-set values. The GPT-3.5 ablation (published Table 6) reports the round-1 model (its values equal V1 in Figure 6), and Table 3's GPT-3.5 values equal the last iteration.

Source: arXiv 2410.12164 v2. Sections 1-5 were read verbatim; the Section 6 tables only as a model summary (the PDF could not be converted here), so the non-SQL numbers below are **unverified**. The summary's NL-to-SQL Llama numbers match the ones verified earlier.

| Task | Validation (dual task) | Test sets (size) | Metric | Llama vanilla → TS (unverified) |
|---|---|---|---|---|
| NL-to-SQL (done) | SQL vs Scala execution agreement | WikiSQL, WikiTQ, Spider, BIRD, Text2Analysis | Exec. acc. | 0.431 → 0.572 (avg of 5) |
| NL-to-Scala | Same pairs; Scala vs SQL agreement | Same 5 | Exec. acc. | 0.176 → 0.343; WikiSQL 0.359 → 0.728, WikiTQ 0.043 → 0.188, Spider 0.249 → 0.356, BIRD 0.100 → 0.228 |
| NL-to-R | R vs second-language agreement | Same 5 | Exec. acc. | 0.260 → 0.333 |
| Schema matching | Permutation consistency of generated Table-B mapping | DeepM 42, WikiData 24, HXD 468 | F1 | 0.839 → 0.873 (WikiData 0.912 → 0.766) |
| Error detection | Permutation consistency of an injected error | Spreadsheet 1,126, Relational 1,081 | F1 | 0.090 → 0.148 |
| Table-QA | Not described in the readable text | FinQA 1,000, TableBench 424, WikiTQ 1,000 | Acc. | 0.270 → 0.296 |
| Data transformation (Pandas / R / SQL) | Execution agreement | TDE 570, Transform-Text 335 | Exec. acc. | 0.113 → 0.142 / 0.057 → 0.090 / 0.057 → 0.078 |

Entity matching, column type annotation and fact verification are listed in Table 1 but have no results.

## Phase 8: error detection, vanilla gate (finished 2026-11-14, 17:55 IST, exit 0)

Llama-3.1-8B-Instruct, repo zero-shot prompt, chat API with the repo system message, greedy, up to 512 tokens. Run IDs `final_ed_instruct_vanilla_{st,rt}_zeroshot`. Cases: the paper's exact counts (see DECISIONS.md).

| AutoTest | Columns (with errors) | Columns flagged | Precision | Recall | **F1 (repo metric)** | F1 (AutoTest rule) | Paper vanilla F1 (unverified) |
|---|---|---|---|---|---|---|---|
| ST-bench ("Spreadsheet-Tables") | 1,126 (42) | 504 | 0.0357 | 0.4286 | **0.0659** | 0.0694 | 0.071 |
| RT-bench ("Relational-Tables") | 1,081 (36) | 342 | 0.0614 | 0.5833 | **0.1111** | 0.1151 | 0.108 |

- **The gate passes:** within 0.006 of the paper on both.
- Instruct flags about 40% of columns, almost all of them clean, so precision is very low; it finds about half of the real errors. 36 and 46 answers had no parseable JSON (counted as no prediction, as in the repo).

## Phase 8: NL-to-Scala smoke test (rerun `smoke_nlscalar2_*`, finished 2026-11-14, 17:55 IST, exit 0)

10 cases per test set, both prompts, vanilla Instruct and Instruct TS seed 1 round 3. Repo-faithful and corrected scores agree on all 16 cells.

| 10 cases each (zero-shot / few-shot) | WikiSQL | Spider | WikiTQ | BIRD |
|---|---|---|---|---|
| Vanilla Instruct | 0.3 / 0.5 | 0.6 / 0.7 | 0.0 / 0.0 | 0.0 / 0.1 |
| Instruct TS r3, seed 1 | 0.7 / 0.7 | 0.9 / 1.0 | 0.4 / 0.3 | 0.2 / 0.2 |

- Vanilla's failures are genuine code errors (checked by hand): `object Main { def main ... }` wrappers that never run when loaded, `finalResult =` without `val`, `createDataFrame` on a bare number. The TS model's code runs.
- **Spark throughput: 0.579 programs/s with 36 workers** (132 programs in 228 s), as estimated.

### Spark throughput with the GPUs busy (2026-11-14, 19:26 IST)

Scoring alone (smoke test): 0.579 programs/s. Scoring while both GPUs generate Scala code (two vLLM servers): 0.56-0.57 programs/s over the first 1,200 programs, about 2% lower, inside the 10% limit set in DECISIONS.md. The jobs run side by side.

### Machine stall, 2026-11-14 19:37 IST to 2026-11-15 03:03 IST (about 7.5 h lost)

**What happened:**
- From about 19:37 IST (14:07 UTC) nothing progressed for about 7.5 hours:
  - **Spark scoring:** wrote no result between 14:15 and 21:34 UTC.
  - **Scala generation:** the BIRD zero-shot cell of TS r3 seed 3 shows a 7 h 28 min gap in its progress bar.
  - **System log:** systemd-journald restarted at 14:13 and 14:21 UTC, reporting a "corrupted or uncleanly shut down" journal.
- Everything resumed by itself at 03:03 IST. Uptime is 9 days, so the machine did not reboot.
- **Likely cause:** the Windows host slept or paused the WSL VM. This is not certain.

**Effects:**
- **36 Spark programs** were in flight during the stall. They were cached as "Sandbox operation timed out" with run times of 26,891-26,955 s, even though the Scala limit is 120 s.
  - These are the only timeouts in the cache, which held 1,758 records at the time.
  - They are not verdicts on the code.
- **Error-detection smoke test:** round 2 failed at 19:55 IST because its vLLM server, started at 19:35 IST, was not ready within 20 minutes. Round 1 had finished correctly beforehand.
- **Scala generation:** the BIRD zero-shot cell of TS r3 seed 3 finished normally after the stall. The client retries failed requests and raises after the last attempt; it never stores an empty answer. Generation is greedy with seed 0, so no stored generation is affected.

**Fix:**
- Cache records that timed out after more than 600 s are now treated as missing and re-run.
- Each re-run is stored beside the original as `<key>.rerun.json`; the original record is kept.
- New test: `test_stalled_timeouts_are_rerun_beside_the_original`. Suite: 190 passed, 1 skipped.
- `nlscala_score1` was stopped before it wrote any cell, then relaunched as `nlscala_score2`. The error-detection smoke test was relaunched as `edts_smoke2`.

## Phase 8: error-detection TS loop, smoke test (rerun `train_edtssmoke2_instruct_seed1`, finished 2026-11-15, 03:28 IST, exit 0)

Seed 1, 2 rounds, 40 tables, 10 negative tables, 5 SFT steps, 20 test columns per benchmark.

| Round | Validator | Typo kept | Semantic kept | Clean kept | Examples | Test F1 (ST, RT) |
|---|---|---|---|---|---|---|
| 1 | vanilla | 4 / 27 | 5 / 33 | 9 / 20 | 18 | 0.0, 0.0 |
| 2 | round 1 adapter | 0 / 24 | 0 / 34 | 20 / 20 | 20 | 0.0, 0.0 |

The pipeline works end to end. The 5-step adapter answers NULL for everything: no prediction on any test column, and as validator it rejects every injected error. This is a smoke-scale artefact, but it marks the risk to watch in the full run: a validator that never flags errors would leave later rounds only clean columns to train on.

## Phase 8: error-detection TS, full runs (running; as of 2026-11-15, 08:30 IST)

Repo metric F1 (AutoTest protocol in brackets). Vanilla from the gate run; paper numbers unverified.

| Model | ST F1 | RT F1 |
|---|---|---|
| Paper vanilla / TS | 0.071 / 0.136 | 0.108 / 0.161 |
| Our vanilla Instruct | 0.0659 | 0.1111 |
| **TS seed 1, round 1** | **0.1397** (0.1531) | **0.1989** (0.2045) |
| **TS seed 2, round 1** | **0.2211** (0.2386) | **0.2836** (0.2923) |
| TS seed 1, round 2 | 0.1954 (0.2179) | 0.2742 (0.2857) |
| TS seed 1, round 3 | 0.2655 (0.2778) | 0.3191 (0.3261) |
| TS seed 2, round 2 | 0.2609 (0.2769) | 0.3009 (0.3091) |
| TS seed 2, round 3 | 0.2909 (0.3168) | 0.2857 (0.3023) |
| TS seed 3, round 1 | 0.1677 (0.1795) | 0.2466 (0.2553) |
| TS seed 3, round 2 | 0.2600 (0.2737) | 0.3265 (0.3333) |
| TS seed 3, round 3 | 0.2791 (0.2927) | 0.2889 (0.2921) |
| **TS round 3, mean of 3 seeds** | **0.2785** | **0.2979** |
| Repair-ED seed 1, round 1 | 0.1545 (0.1700) | 0.2353 (0.2454) |
| Repair-ED seed 1, round 2 | 0.3077 (0.3509) | 0.3366 (0.3542) |
| Repair-ED seed 1, round 3 | 0.2459 (0.2609) | 0.2752 (0.2885) |
| Best-of-2 control seed 1, round 1 | 0.1311 (0.1404) | 0.1834 (0.1867) |

Seed 1 kept, by round (typo / semantic / clean, of candidates):
- Round 1: 477 / 2,007, 663 / 2,569, 1,048 / 2,000. 2,188 examples.
- Round 2: 491 / 1,983, 1,393 / 2,573, 1,653 / 2,000. 3,537 examples.
- Round 3: 541 / 1,993, 1,649 / 2,580, 1,746 / 2,000. 3,936 examples.

Seed 2 round 3: kept 698 / 2,056 typo, 1,793 / 2,569 semantic, 1,786 / 2,000 clean, 4,277 examples (300 steps, 3.1 h of training). Seed 2 finished 2026-11-15, 20:39 IST, exit 0.

Seed 3 (finished 2026-11-16, 04:32 IST, exit 0): 2,256, then 3,531, then 3,945 examples. Like seed 2, ST rises every round and RT dips in round 3 (0.3265 to 0.2889). Round 3 means over 3 seeds: ST 0.2785, RT 0.2979 (paper TS 0.136, 0.161).

Repair-ED seed 1 round 1 (2026-11-16 about 01:15 IST): of 3,436 rejected errors (1,530 typo, 1,906 semantic), 3,056 retries were usable and 506 passed Algorithm 2 (178 typo, 328 semantic), so 2,694 examples against TS's 2,188. F1 ST 0.1545 and RT 0.2353, against TS seed 1 round 1's 0.1397 and 0.1989. One seed; the controls will show whether the gain comes from the feedback or from the extra examples.

Repair-ED seed 1 finished 2026-11-16 about 08:00 IST: round 2 recovered 491 (3,954 examples; F1 ST 0.3077, RT 0.3366, the best single ED adapter so far, against TS seed 1 round 2's 0.1954 and 0.2742); round 3 recovered 653 (4,452 examples) but F1 fell to ST 0.2459 and RT 0.2752, below TS seed 1 round 3 (0.2655, 0.3191). Best-of-2 control seed 1 (07:40 IST): 446 recovered without feedback (2,634 examples), F1 ST 0.1311 and RT 0.1834, below both TS round 1 and repair-ED round 1. One seed each; seeds 2-3 running.

Each round's adapter is a more lenient validator than the last, so more columns are kept. Test F1 rises every round for seed 1. Seed 2 rises on ST every round, but RT dips in round 3 (0.3009 to 0.2857 repo metric). Seed 1 finished 2026-11-15, 17:26 IST, exit 0.

Seed 1 round 1:
- **Kept:** 477 of 2,007 typo errors, 663 of 2,569 semantic errors and 1,048 of 2,000 clean columns, for 2,188 training examples (300 steps).
- **ST:** precision 0.0856, recall 0.381, a prediction on 187 of 1,126 columns.
- **RT:** precision 0.1241, recall 0.500, a prediction on 145 of 1,081 columns.
- **Collapse check:** the adapter still flags errors, so the collapse seen in the smoke test did not happen at full scale.
- **Timings:** generation 14 min, validation 74 min, training 2 h 38 min, test 7 min.

## Phase 8: NL-to-Scala, final rounds, zero-shot (finished 2026-11-15, 14:08 IST, exit 0)

Llama-3.1-8B-Instruct, mean ± sample SD over 3 seeds. The **repo-faithful score** is the repo's `nss_evaluator`: exact, order-sensitive comparison. The corrected score is in brackets: order-insensitive unless the gold SQL has ORDER BY, and escape-aware CSV reading. Paper numbers are unverified (read from a summary).

| Model | WikiSQL | WikiTQ | Spider | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Paper vanilla | 0.359 | 0.043 | 0.249 | 0.100 | 0.1878 |
| Paper TS | 0.728 | 0.188 | 0.356 | 0.228 | 0.3750 |
| Our vanilla | 0.3410 (0.3410) | 0.0420 (0.0420) | 0.3005 (0.3815) | 0.0909 (0.0937) | 0.1936 (0.2146) |
| Our TS r3 | 0.6710 ± 0.0453 (0.6710) | 0.3150 ± 0.0217 (0.3160) | 0.5765 ± 0.0019 (0.6639) | 0.3039 ± 0.0069 (0.3076) | 0.4666 (0.4896) |
| Our repair-TS r3 | 0.6803 ± 0.0265 (0.6803) | 0.3370 ± 0.0142 (0.3380) | 0.6247 ± 0.0025 (0.7234) | 0.2847 ± 0.0105 (0.2865) | 0.4817 (0.5071) |

- **Vanilla reproduces the paper:** within 0.018 on WikiSQL, WikiTQ and BIRD; Spider is +0.052.
- **TS r3 is far above the paper's TS** on WikiTQ (+0.127), Spider (+0.221) and BIRD (+0.076), and 0.057 below on WikiSQL. Mean of 4: 0.4666 against 0.3750.
- **The paper's own figures disagree:** the summary gives the paper's Llama Scala average as 0.343, but the mean of its four numbers is 0.375. This is one more reason to treat the paper's figures as unverified. *(Corrected 2026-11-18: 0.343 is the 5-set average including Text2Analysis, 0.214; the published Table 3 confirms all four values. No disagreement.)*
- **Repair-TS vs TS:** +0.0151 mean of 4. Per-seed means of 4: repair-TS 0.4781, 0.4834, 0.4835; TS 0.4813, 0.4575, 0.4610. Repair-TS is higher on 2 of 3 seeds. Its gains are on WikiTQ (+0.022), Spider (+0.048) and WikiSQL (+0.009); it is lower on BIRD (−0.019). Paired bootstrap: +0.0150 [+0.0079, +0.0221], p < 0.0001 (see "all 19 models by round" below).
- **Few-shot:** finished; see the next section.

## Phase 8: NL-to-Scala, final rounds, few-shot (finished 2026-11-15, 18:16 IST, exit 0)

Same models, cases and scoring as the zero-shot table; few-shot prompt from the repo's `nl2scala_prompt`. Mean ± sample SD over 3 seeds. The paper reports zero-shot only for Scala (unverified).

| Model | WikiSQL | WikiTQ | Spider | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| vanilla (repo) | 0.6720 | 0.1290 | 0.3907 | 0.1625 | 0.3386 |
| vanilla (corrected) | 0.6720 | 0.1300 | 0.4766 | 0.1625 | 0.3603 |
| TS r3 (repo) | 0.6630 ± 0.0275 | 0.2977 ± 0.0046 | 0.5802 ± 0.0131 | 0.2755 ± 0.0138 | 0.4541 |
| TS r3 (corrected) | 0.6630 ± 0.0275 | 0.2987 ± 0.0046 | 0.6725 ± 0.0152 | 0.2773 ± 0.0152 | 0.4779 |
| repair-TS r3 (repo) | 0.6727 ± 0.0234 | 0.3287 ± 0.0211 | 0.6116 ± 0.0076 | 0.2773 ± 0.0042 | 0.4726 |
| repair-TS r3 (corrected) | 0.6727 ± 0.0234 | 0.3297 ± 0.0211 | 0.7095 ± 0.0109 | 0.2800 ± 0.0064 | 0.4980 |

## Phase 8: NL-to-Scala, all 19 models by round (scoring finished 2026-11-16, 06:04 IST, exit 0)

All 152 cells are scored (19 models x 4 test sets x 2 prompts). Repo score per test set (mean over seeds 1-3); mean of 4 as mean ± SD over seeds; corrected score in the last column. From `python -m ts.analysis.scala_table`.

Zero-shot:

| Model | WikiSQL | WikiTQ | Spider | BIRD | Mean of 4 (repo) | Per seed | Corrected |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.3410 | 0.0420 | 0.3005 | 0.0909 | 0.1936 | | 0.2146 |
| TS r1 | 0.7137 | 0.1743 | 0.4477 | 0.2085 | 0.3860 ± 0.0093 | 0.3958, 0.3850, 0.3773 | 0.4100 ± 0.0094 |
| TS r2 | 0.6963 | 0.2620 | 0.5476 | 0.2580 | 0.4410 ± 0.0298 | 0.4576, 0.4589, 0.4066 | 0.4654 ± 0.0292 |
| TS r3 | 0.6710 | 0.3150 | 0.5765 | 0.3039 | 0.4666 ± 0.0129 | 0.4813, 0.4575, 0.4610 | 0.4896 ± 0.0134 |
| Repair-TS r1 | 0.7040 | 0.2037 | 0.4752 | 0.2231 | 0.4015 ± 0.0174 | 0.4204, 0.3980, 0.3861 | 0.4264 ± 0.0179 |
| Repair-TS r2 | 0.7170 | 0.3227 | 0.6016 | 0.2883 | 0.4824 ± 0.0098 | 0.4936, 0.4753, 0.4783 | 0.5075 ± 0.0103 |
| Repair-TS r3 | 0.6803 | 0.3370 | 0.6247 | 0.2847 | 0.4817 ± 0.0031 | 0.4781, 0.4834, 0.4835 | 0.5071 ± 0.0035 |

Few-shot:

| Model | WikiSQL | WikiTQ | Spider | BIRD | Mean of 4 (repo) | Per seed | Corrected |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.6720 | 0.1290 | 0.3907 | 0.1625 | 0.3386 | | 0.3603 |
| TS r1 | 0.6823 | 0.2043 | 0.4942 | 0.2029 | 0.3959 ± 0.0187 | 0.4117, 0.4009, 0.3753 | 0.4192 ± 0.0182 |
| TS r2 | 0.6827 | 0.2690 | 0.5465 | 0.2461 | 0.4361 ± 0.0229 | 0.4577, 0.4383, 0.4122 | 0.4601 ± 0.0222 |
| TS r3 | 0.6630 | 0.2977 | 0.5802 | 0.2755 | 0.4541 ± 0.0083 | 0.4545, 0.4456, 0.4621 | 0.4779 ± 0.0089 |
| Repair-TS r1 | 0.6940 | 0.2137 | 0.5086 | 0.2186 | 0.4087 ± 0.0108 | 0.4205, 0.3991, 0.4066 | 0.4332 ± 0.0106 |
| Repair-TS r2 | 0.6910 | 0.3107 | 0.5904 | 0.2562 | 0.4621 ± 0.0174 | 0.4819, 0.4494, 0.4549 | 0.4862 ± 0.0187 |
| Repair-TS r3 | 0.6727 | 0.3287 | 0.6116 | 0.2773 | 0.4726 ± 0.0044 | 0.4758, 0.4675, 0.4743 | 0.4980 ± 0.0039 |

TS improves every round, with both prompts. Repair-TS is ahead of TS at every round with both prompts. In the zero-shot mean of 4 it is reached by round 2 (0.4824) and flat in round 3 (0.4817); its seed spread in round 3 is a quarter of TS's (SD 0.0031 against 0.0129).

Paired bootstrap (`python -m ts.analysis.significance --task scala`, written to `results/analysis/significance_scala.json`; per case, correctness averaged over seeds; mean of 8 cells with equal weights; 10,000 draws):

| Comparison | Mean of 8: diff [95% CI], p | Zero-shot mean of 4: diff [95% CI], p | Per seed (mean of 8) |
|---|---|---|---|
| TS r3 vs vanilla | +0.1943 [+0.1842, +0.2045], p < 0.0001 | +0.2730 [+0.2576, +0.2885], p < 0.0001 | +0.2018, +0.1855, +0.1955 |
| Repair-TS r3 vs vanilla | +0.2110 [+0.2007, +0.2214], p < 0.0001 | +0.2881 [+0.2727, +0.3034], p < 0.0001 | +0.2109, +0.2094, +0.2128 |
| TS r3 vs TS r1 | +0.0693 [+0.0621, +0.0766], p < 0.0001 | +0.0806 [+0.0698, +0.0917], p < 0.0001 | +0.0642, +0.0586, +0.0852 |
| Repair-TS r1 vs TS r1 | +0.0141 [+0.0098, +0.0183], p < 0.0001 | +0.0155 [+0.0094, +0.0215], p < 0.0001 | +0.0167, +0.0056, +0.0200 |
| Repair-TS r2 vs TS r2 | +0.0337 [+0.0284, +0.0389], p < 0.0001 | +0.0414 [+0.0333, +0.0493], p < 0.0001 | +0.0301, +0.0138, +0.0572 |
| Repair-TS r3 vs TS r3 | +0.0167 [+0.0119, +0.0216], p < 0.0001 | +0.0150 [+0.0079, +0.0221], p < 0.0001 | +0.0091, +0.0239, +0.0173 |

Every per-seed difference is positive. The case bootstrap does not cover seed-to-seed variation; with 3 seeds, all positive at every round, the seeds agree with it.

### WSL crash, 2026-11-15 about 12:45 IST

The crash was Wsl/Service/E_UNEXPECTED; the user ran `wsl --shutdown`. Every job was killed. The last log lines are from 07:11 UTC (12:41 IST).

**State of each job:**

| Job | Done | Interrupted | On resume |
|---|---|---|---|
| ED seed 1 | round 1 (summary.json) | round 2 SFT at step 210 of 306 (`adapter_ed/checkpoint-210`; loads, optimizer, scheduler and RNG state intact) | training continues from step 210 (96 steps, about 1.3 h), then the test and round 3 |
| ED seed 2 | round 1 | round 2: errors.jsonl complete (5,594 rows), validation not started | validation from the start (about 1.5 h), training, test, round 3 |
| ED seed 3 | none | round 1: errors.jsonl complete (5,596 rows), validation at 3,143 of 32,820 requests | validation again; the 3,143 finished requests come from the response cache |
| Scala zero-shot `nlscala_score2` | 15,420 of 15,420 programs; 17 of 27 cells written | writing cells. repair-TS r3 s1 WikiTQ: partial `preds.jsonl`, 959 of 1,000 lines, no result file | the 10 unwritten cells plus TS r3 s3 BIRD (363 cases); the partial `preds.jsonl` is rewritten, so it is backed up first |
| Scala few-shot `nlscala_score3` | about 2,600 of 12,087 programs cached | Spark execution | the rest; cached programs are not run again |

**Integrity checks:**
- All execution-cache records (20,062) and result files parse. No `.tmp` files are left over.
- `runs/llm_cache.sqlite`: quick_check ok.
- The flock-based GPU and Spark locks leave nothing stale behind.
- 36 stopped sandbox containers are left over.

**Round 1 of seed 2:** F1 ST 0.2211 (AutoTest 0.2386), RT 0.2836 (0.2923). Kept 513 typo errors, 727 semantic errors and 1,056 clean columns.

**Writing cells was slow for TS BIRD (fixed 2026-11-15, 18:05 IST).** It took about 43 min per TS BIRD cell, against about 1 min for vanilla. The cause was a quadratic row-count difference in `compare._unordered_equal`; TS reaches it often because its results have the gold's row count. After the fix, a BIRD cell takes about 2 s. Verdicts are identical: 45,930 of 45,930 cases rescored read-only, 0 differences (DECISIONS.md).

## Complete per-dataset table

Every approach on every test set, both models and both prompts: `results/analysis/results_table.md`, regenerated by `python -m ts.analysis.results_table > results/analysis/results_table.md`. Paired significance tests: `results/analysis/significance.json` (`python -m ts.analysis.significance`).

## Compute estimate

Measured figures from round 1 replace all earlier estimates, including the 161-hour sequential estimate, which assumed 30,000 Spark programs per round. The run actually ran 6,172.

| Arm, 3 seeds × 3 rounds, one GPU each | Time per round | Total | Projected finish, IST |
|---|---|---|---|
| Table-Specialist | 4.57 h in round 1, 5.59 h in round 2; about 6 h assumed for round 3 | About 49 h | Fri 6 Nov, around midnight |
| No-validation | 7.86 h measured in round 1 | About 71 h | Sat 7 Nov, about 22:30-23:30 |

Both finishes assume no further stops. The study started on Wed 4 Nov 2026 at about 23:25 IST.

### Error detection: same-data control, round 1 (2026-11-16, 14:39 IST)

TS round 1 kept columns plus extra kept errors from the donor seed's TS round 1, on tables this seed never used; as many of each kind as repair-ED round 1 recovered. Repo-faithful F1, ST / RT. Log `logs/edcontrols.log`.

| Seed | Donor | Examples | ST | RT |
|---|---|---|---|---|
| 1 | 2 | 2,694 (2,188 kept + 506 donor) | 0.1561 | 0.2280 |
| 2 | 3 | 2,787 (2,296 kept + 491 donor) | 0.2404 | 0.2258 |
| 3 | 1 | 2,774 (2,256 kept + 518 donor) | 0.1792 | 0.2532 |

### Repair-ED, all finished rounds (2026-11-16, 20:15 IST)

Repo-faithful F1, ST / RT. TS 3-seed mean at round 3: 0.2785 / 0.2979.

| Seed | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| 1 | 0.1545 / 0.2353 | 0.3077 / 0.3366 | 0.2459 / 0.2752 |
| 2 | 0.2429 / 0.3303 | 0.2430 / 0.3333 | 0.2558 / 0.3768 |
| 3 | 0.2034 / 0.2553 | 0.1916 / 0.2439 | 0.2742 / 0.3673 |
| Mean | 0.2003 / 0.2736 | 0.2474 / 0.3046 | 0.2586 / 0.3398 |

Repair-ED finished 2026-11-17 01:17 IST, exit 0. Round 3 mean against TS round 3 mean: ST -0.0199, RT +0.0419 (significance tests pending).

### NL-to-R TS, seed 1 (training data)

| Round | Kept of questions | agree / disagree / r_failed / always_empty / sql_failed / no_code |
|---|---|---|
| 1 | 321 of 2,978 | 321 / 661 / 640 / 977 / 217 / 159 |
| 2 | 525 of 2,976 | 525 / 577 / 551 / 1,035 / 187 / 100 |

### Error detection: paired bootstrap tests (2026-11-17, 08:35 IST)

`python -m ts.analysis.ed_significance`, `results/analysis/significance_ed.json`. Columns resampled within each test set (10,000 draws, same draw for both models and all seeds), F1 recomputed per draw as in `ed_eval.metric`; statistic = mean over ST and RT of the seed-mean F1 difference. Point F1s reproduce the scorer's values exactly.

| Comparison | Mean of ST, RT | 95% CI | p | ST | RT | Per seed |
|---|---|---|---|---|---|---|
| TS r3 vs vanilla | +0.1997 | [0.1347, 0.2613] | <0.0001 | +0.2126 | +0.1868 | +0.204, +0.200, +0.196 |
| Repair-ED r3 vs vanilla | +0.2107 | [0.1433, 0.2733] | <0.0001 | +0.1927 | +0.2287 | +0.172, +0.228, +0.232 |
| TS r3 vs TS r1 | +0.0786 | [0.0347, 0.1200] | 0.0002 | +0.1023 | +0.0549 (p 0.10) | +0.123, +0.036, +0.077 |
| Repair-ED r1 vs TS r1 | +0.0274 | [0.0030, 0.0516] | 0.027 | +0.0241 | +0.0306 | +0.026, +0.034, +0.022 |
| Repair-ED r2 vs TS r2 | +0.0064 | [-0.0260, 0.0380] | 0.70 | +0.0087 | +0.0041 | +0.087, +0.007, -0.076 |
| Repair-ED r3 vs TS r3 | +0.0110 | [-0.0261, 0.0502] | 0.58 | -0.0199 | +0.0419 | -0.032, +0.028, +0.037 |
| Best-of-2 r1 vs TS r1 | +0.0215 | [-0.0052, 0.0492] | 0.12 | +0.0156 | +0.0274 | -0.012, +0.028, +0.049 |
| Same-data r1 vs TS r1 | +0.0042 | [-0.0161, 0.0251] | 0.69 | +0.0158 | -0.0074 | +0.023, -0.019, +0.009 |
| Repair-ED r1 vs best-of-2 r1 | +0.0058 | [-0.0232, 0.0361] | 0.71 | +0.0085 | +0.0032 | +0.038, +0.007, -0.027 |
| Repair-ED r1 vs same-data r1 | +0.0232 | [0.0024, 0.0433] | 0.031 | +0.0083 | +0.0380 (p 0.009) | +0.003, +0.053, +0.013 |

Reading: TS's error-detection gain over vanilla is large and certain. Repair-ED helps at round 1 (+0.027, all 3 seeds positive, p 0.03) and beats the same-data control (p 0.03), so its round-1 gain is not data volume alone; but it does not beat best-of-2 sampling (p 0.71), so a second attempt explains it as well as the feedback does. By rounds 2-3 the gain is gone (p 0.70, 0.58; seeds disagree in sign). Error-detection test sets are small (42 and 36 columns with errors), so intervals are wide.

### Error detection: complete results table (2026-11-17, 08:50 IST)

Repo-faithful F1, Llama-3.1-8B-Instruct, zero-shot. ST = Spreadsheet-Tables (1,126 columns, 42 with errors), RT = Relational-Tables (1,081, 36). Paper figures unverified (read from a summary of the PDF). Mean ± sample SD over seeds 1-3; avg = mean of ST and RT.

| Model | Round | ST seed 1 / 2 / 3 | ST mean ± SD | RT seed 1 / 2 / 3 | RT mean ± SD | Avg |
|---|---|---|---|---|---|---|
| Paper vanilla | - | - | 0.071 | - | 0.108 | 0.090 |
| Paper TS | final | - | 0.136 | - | 0.161 | 0.148 |
| Our vanilla | - | - | 0.0659 | - | 0.1111 | 0.0885 |
| Our TS | 1 | 0.1397 / 0.2211 / 0.1677 | 0.1762 ± 0.0413 | 0.1989 / 0.2836 / 0.2466 | 0.2430 ± 0.0425 | 0.2096 |
| Our TS | 2 | 0.1954 / 0.2609 / 0.2600 | 0.2388 ± 0.0375 | 0.2742 / 0.3009 / 0.3265 | 0.3005 ± 0.0262 | 0.2696 |
| Our TS | 3 | 0.2655 / 0.2909 / 0.2791 | 0.2785 ± 0.0127 | 0.3191 / 0.2857 / 0.2889 | 0.2979 ± 0.0185 | 0.2882 |
| Repair-ED | 1 | 0.1545 / 0.2429 / 0.2034 | 0.2003 ± 0.0442 | 0.2353 / 0.3303 / 0.2553 | 0.2736 ± 0.0501 | 0.2369 |
| Repair-ED | 2 | 0.3077 / 0.2430 / 0.1916 | 0.2474 ± 0.0582 | 0.3366 / 0.3333 / 0.2439 | 0.3046 ± 0.0526 | 0.2760 |
| Repair-ED | 3 | 0.2459 / 0.2558 / 0.2742 | 0.2586 ± 0.0144 | 0.2752 / 0.3768 / 0.3673 | 0.3398 ± 0.0561 | 0.2992 |
| Best-of-2 control | 1 | 0.1311 / 0.2156 / 0.2286 | 0.1918 ± 0.0529 | 0.1834 / 0.3443 / 0.2837 | 0.2705 ± 0.0812 | 0.2311 |
| Same-data control | 1 | 0.1561 / 0.2404 / 0.1792 | 0.1919 ± 0.0436 | 0.2280 / 0.2258 / 0.2532 | 0.2357 ± 0.0152 | 0.2138 |

## Phase 9: NL-to-R test results, vanilla and TS-R (2026-11-17, 13:50 IST)

Llama-3.1-8B-Instruct, repo `nr_evaluator` rules (repo-faithful), mean over seeds 1-3 (SD of the mean of 4 over seeds). 7,122 cases per model. Paper NL-to-R (unverified, average of 5 sets incl. Text2Analysis): vanilla 0.260, TS 0.333; per-set paper values not available. *(Corrected 2026-11-18: the published Table 3 gives them: 0.331/0.409, 0.138/0.257, 0.503/0.536, 0.225/0.261 for WikiSQL, WikiTQ, Spider, BIRD; mean of 4 0.2993/0.3658.)*

| Model | Prompt | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|---|
| Vanilla | zero-shot | 0.3260 | 0.5993 | 0.1500 | 0.2314 | 0.3267 |
| Vanilla | few-shot | 0.3830 | 0.6152 | 0.2220 | 0.2782 | 0.3746 |
| TS-R r1 | zero-shot | 0.4147 | 0.6939 | 0.2817 | 0.3453 | 0.4339 ± 0.0193 |
| TS-R r2 | zero-shot | 0.4560 | 0.7101 | 0.2977 | 0.3673 | 0.4578 ± 0.0170 |
| TS-R r3 | zero-shot | 0.5057 | 0.7156 | 0.3147 | 0.3682 | 0.4760 ± 0.0241 |
| TS-R r1 | few-shot | 0.4357 | 0.6714 | 0.2803 | 0.3297 | 0.4293 ± 0.0096 |
| TS-R r2 | few-shot | 0.5043 | 0.6853 | 0.3087 | 0.3489 | 0.4618 ± 0.0134 |
| TS-R r3 | few-shot | 0.5420 | 0.7012 | 0.3317 | 0.3499 | 0.4812 ± 0.0304 |

Per seed, zero-shot mean of 4 (r1 / r2 / r3): seed 1 0.4547 / 0.4737 / 0.5024; seed 2 0.4304 / 0.4599 / 0.4551; seed 3 0.4166 / 0.4398 / 0.4706.

Paired bootstrap (`python -m ts.analysis.significance --task r`, `results/analysis/significance_r.json`), mean of 8: TS-R r1 vs vanilla +0.0809, r2 +0.1091, r3 +0.1280 (all p < 0.0001, every seed positive); TS-R r3 vs r1 +0.0470 (p < 0.0001, every seed positive). Unlike Instruct NL-to-SQL, rounds help on R.

## Phase 9: NL-to-R complete (2026-11-17, 22:46 IST)

All 152 cells scored (vanilla + 9 TS-R + 9 repair-TS-R models x 8 cells), every run exit 0. Repo-faithful accuracy, mean over seeds 1-3; mean of 4 ± SD over seeds.

| Model | Prompt | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 | Seeds 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| Paper vanilla / TS (published Table 3; avg of 5 was 0.260 / 0.333) | zero-shot | 0.331 / 0.409 | 0.503 / 0.536 | 0.138 / 0.257 | 0.225 / 0.261 | 0.2993 / 0.3658 | |
| Vanilla | zero-shot | 0.3260 | 0.5993 | 0.1500 | 0.2314 | 0.3267 | |
| TS-R r1 | zero-shot | 0.4147 | 0.6939 | 0.2817 | 0.3453 | 0.4339 ± 0.0193 | 0.4547 / 0.4304 / 0.4166 |
| TS-R r2 | zero-shot | 0.4560 | 0.7101 | 0.2977 | 0.3673 | 0.4578 ± 0.0170 | 0.4737 / 0.4599 / 0.4398 |
| TS-R r3 | zero-shot | 0.5057 | 0.7156 | 0.3147 | 0.3682 | 0.4760 ± 0.0241 | 0.5024 / 0.4551 / 0.4706 |
| Repair-TS-R r1 | zero-shot | 0.4270 | 0.7087 | 0.2977 | 0.3407 | 0.4435 ± 0.0108 | 0.4541 / 0.4440 / 0.4325 |
| Repair-TS-R r2 | zero-shot | 0.5107 | 0.7212 | 0.3230 | 0.3673 | 0.4805 ± 0.0103 | 0.4884 / 0.4844 / 0.4688 |
| Repair-TS-R r3 | zero-shot | 0.6023 | 0.7159 | 0.3333 | 0.3655 | **0.5043 ± 0.0128** | 0.5182 / 0.4930 / 0.5016 |
| Vanilla | few-shot | 0.3830 | 0.6152 | 0.2220 | 0.2782 | 0.3746 | |
| TS-R r1 | few-shot | 0.4357 | 0.6714 | 0.2803 | 0.3297 | 0.4293 ± 0.0096 | 0.4399 / 0.4269 / 0.4211 |
| TS-R r2 | few-shot | 0.5043 | 0.6853 | 0.3087 | 0.3489 | 0.4618 ± 0.0134 | 0.4739 / 0.4642 / 0.4473 |
| TS-R r3 | few-shot | 0.5420 | 0.7012 | 0.3317 | 0.3499 | 0.4812 ± 0.0304 | 0.5120 / 0.4512 / 0.4803 |
| Repair-TS-R r1 | few-shot | 0.4607 | 0.6839 | 0.2950 | 0.3269 | 0.4416 ± 0.0045 | 0.4463 / 0.4374 / 0.4413 |
| Repair-TS-R r2 | few-shot | 0.5413 | 0.7031 | 0.3297 | 0.3434 | 0.4794 ± 0.0055 | 0.4782 / 0.4854 / 0.4746 |
| Repair-TS-R r3 | few-shot | 0.6087 | 0.7076 | 0.3410 | 0.3499 | 0.5018 ± 0.0195 | 0.5236 / 0.4860 / 0.4957 |

Paired bootstrap, mean of 8 (`results/analysis/significance_r.json`): repair-TS-R vs TS-R +0.0110 (r1), +0.0202 (r2), +0.0244 (r3), all p < 0.0001, every seed positive at every round; zero-shot mean of 4 +0.0096 (p 0.0024), +0.0228, +0.0282. The gain grows with rounds and comes mostly from WikiSQL (+0.097 zero-shot at r3) and WikiTQ; Spider and BIRD are flat. No same-data control was run for R, so repair's effect is not separated from the extra 16-31% of training pairs (25-31% in round 1, 16-23% in rounds 2-3; corrected 2026-11-18 from run summaries).
