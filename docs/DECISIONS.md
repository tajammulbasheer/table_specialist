# Decisions

Last updated: 2026-11-04 12:10 (machine clock).

Each entry gives the choice, why it was made, and whether it comes from the **paper**, the **repo** or **us**. Anything marked "us" is a difference to report, because the paper does not publish it. When a decision changes, the old reasoning stays here.

Sources: [paper](https://arxiv.org/abs/2410.12164), [repo at `af1921f`](https://github.com/microsoft/Table-Specialist), and the [Blueprint](https://claude.ai/artifact/VwKgdw6sfVGXyRLJt8A5cw).

## Reversed decisions

The deadline was removed on 2026-11-04, so choices made to save time or memory were reopened.

| Topic | Was | Now | Why it was chosen | Why it changed |
|---|---|---|---|---|
| Training method | QLoRA (4-bit weights) | LoRA, as the paper says | The Blueprint estimated BF16 LoRA at 20-23 GB, "borderline" on a 24 GB card, while QLoRA fitted with margin | Closer to the paper. Measured: 20.0 GB at about 4,000 tokens, at about the same speed as QLoRA |
| Second language | Pandas | Spark-Scala, as the paper says | The repo has an NL-to-Pandas generator and a Python sandbox, and Spark needed no setup | Closer to the paper. The repo has the Scala prompt and sandbox. Measured cost: about 11 hours per round at 36 workers, against 0.8 hours assumed for Pandas. We accept this because the user removed the deadline |
| Rounds | 2 | 3, as the paper says ("up to 3") | Cut for the deadline | No deadline |
| Scope | Error Detection and ablations only if time remained | Back in the plan, after the core study | Cut for the deadline | No deadline |
| WikiSQL size | 1,000 sampled, then 975 after the prompt limit | Exactly 1,000 that fit the limit | First implementation | The paper reports 1,000 |
| Llama request | No system message, then the repo's system message, both through chat | Raw completions, no system message | The repo's chat call sends a system message | Base has no chat template, so chat cannot be used. The repo's call is written for GPT-3.5; its Llama call is unpublished |
| Model checkpoint | Llama-3.1-8B-Instruct | **Llama-3.1-8B (base)** | The Blueprint chose Instruct, and I then defended it because its scores were nearer the paper's | The user directed this on 2026-11-04: reproduce faithfully and keep Instruct as a future experiment. The paper says only "Llama-3.1-8B" and the repo's README links the base checkpoint. Choosing a model by how close its score lands to the paper was the wrong test |

## Baseline gate outcome

Decided 2026-11-04: **proceed with the gate not passed, and record the gaps as unexplained.** Base meets ± 0.05 only on BIRD. Six setting combinations were tested on WikiSQL or more, none reproduces the paper, and more combinations would eventually land near 0.525 by chance. What this means for the report: compare the ordering of the three arms, not the absolute scores, and state the vanilla gaps next to every result.

| Decision | Source | Why |
|---|---|---|
| Evaluate with the repo's zero-shot prompt, lowercase WikiSQL tables, and report few-shot scores beside it | Repo | `main_build_data.py` builds zero-shot test prompts by default, and the repo's lowercased gold SQL needs lowercase data. **This was not chosen for closeness to the paper:** few-shot is closer on 3 of 4 benchmarks |

## Models

| Decision | Source | Why |
|---|---|---|
| Llama-3.1-8B **base** | Paper and repo | The paper says "Llama-3.1-8B", and the repo's README links `meta-llama/Llama-3.1-8B`. "Vanilla" in the paper means not fine-tuned; it does not name a checkpoint |
| Instruct is a future experiment, not part of this reproduction | User | Directed on 2026-11-04. Its earlier scores stay in `docs/RESULTS.md`, labelled as such |
| Pin base to `d04e592`, Instruct to `0e9e39f` | Us | The paper gives no revision |
| Qwen only for pipeline testing | Us | It is not in the paper's Llama results |
| Benchmarks use the repo's zero-shot test prompt, unchanged | Repo | `main_build_data.py` builds test prompts this way by default. **Unconfirmed for the paper:** its Appendix A shows a four-example NL-to-SQL prompt without saying whether evaluation used it, and this is being tested |
| Generation uses the few-shot prompts read directly from the repo's scripts | Repo | `ts/gen/prompts.py` reads them from the syntax tree, so they stay byte for byte. The fixed text is about 2,150 tokens, so the server needs more than 4,096 tokens of context |
| Raw completions, keeping the first SQL block, at most 128 output tokens | Us | Base has no chat template, keeps writing after its answer, and the repo's parser takes the last block. The authors' Llama call is unpublished, so this may differ from theirs |
| Temperature 0 for code, at most 256 output tokens | Repo default (0), us (tokens) | The paper does not state a temperature |

## Test sets: paper against ours

| Benchmark | Paper | Ours | Match | Why this way |
|---|---|---|---|---|
| WikiSQL | 1,000 | Test split; first 1,000 of a shuffle with seed 42 that fit the prompt limit | Count only; the cases are probably different | Paper's sampling is unpublished |
| Spider | 1,198 | Test split; gold SQL touching one table (1,230); prompt limit removes 32 | **Count matches exactly.** Unconfirmed that the cases are the same | Dev has only 575 single-table questions, so it cannot be dev |
| WikiTQ | 1,000 | `pristine-unseen-tables` (the dataset's test split); same sampling as WikiSQL | Count only | Paper's sampling is unpublished |
| BIRD | 356 | Dev; gold SQL touching one table | **No: 363, 7 more than the paper.** Cause unknown. No single check gives 7: empty gold 0, NULL gold 1, gold that fails 0, empty tables 0, JOIN in the gold SQL 1, duplicate gold SQL 3 | We do not delete cases to force the count |
| Text2Analysis | 271 | None | **Not possible** | Its README, checked 2026-11-04: available "after the company's approval process" |

Averages across benchmarks must leave out Text2Analysis on both sides.

AutoTest, for Error Detection after the core study: each file has 1,200 columns against the paper's 1,126 and 1,081. Removing debatable labels gives 1,138 and 1,170, which does not match, so the paper's filter is unknown.

**Resolved 2026-11-14:** the filter is the repo's prompt-length limit (as for the SQL test sets). With the repo's `ed` prompt (EDPromptGenerator, markdownvalueonly, 70% of 4,096 tokens), ST-bench keeps exactly **1,126** columns (73 too long, 1 row whose value list cannot be parsed) and RT-bench exactly **1,081** (119 too long): the paper's counts. Columns with errors among them: 42 (ST) and 36 (RT). `ts/eval/ed_eval.py` uses these cases.

### Other test-set choices

| Decision | Source | Why |
|---|---|---|
| Lowercase WikiSQL cells in both the prompt and the database | Us | WikiSQL's `test.db` stores them lowercase, and the repo lowercases gold SQL, which needs lowercase data. The paper is silent. With Instruct on the earlier 975-case run, 97 of 436 misses contained a capitalised value, so this may cost score. On base, showing original case instead lowered WikiSQL to 0.142 (zero-shot) and 0.287 (few-shot), so it does not explain the gap |
| Show the first 20, 10, 5 or 3 rows of a BIRD table; run SQL on the whole table | Us, following the repo | Tables are too large for a prompt. The repo's evaluator reads BIRD's database and CSV separately. The row counts are ours |
| Do not show BIRD's evidence strings | Repo | The repo's prompt uses only the question. The paper is silent |
| Rename each table to `"table"` and rewrite gold SQL to match | Repo | The repo's prompt tells the model the table is called "table" |

## Training tables: why GitTables and not the original

| Decision | Source | Why |
|---|---|---|
| GitTables 1M licensed subset instead of the paper's corpus | Us | **The paper's corpus is unreleased.** The paper says "a large corpus" |
| Keep web tables as a later comparison | Us | The repo's filenames (`table2question_all_webtable_v4_1st_3K`) suggest the authors used web tables. Those can overlap WikiSQL and WikiTQ, so they need the contamination filter first |
| Never train on benchmark splits | Paper | The paper: "we do not use the training split of any benchmark data during fine-tuning" |
| 5-30 columns, 10-200 rows, up to 2,000 tokens, header similarity measured as Jaccard at 0.9 | Repo (5+ columns), us (rest) | Unpublished |
| Corpus of 10,053 tables instead of the planned 20,000 | Us | That is all these filters give at the limit below. The study needs 9,000 per seed (3 rounds of 3,000) |
| Treat headers differing only by case as duplicates | Us | SQLite ignores case in column names, so `Logon` and `logon` cannot share a table; this crashed a real run. Corpus went from 10,378 to 10,053 tables |
| Keep at most 5 tables per near-identical header set | Us | Measured kept tables at limits 1, 2, 3, 5 and 10: 5,036, 7,430, 8,751, 10,378, 12,640. 5 is the smallest that reaches 9,000. Chosen on table counts only, before any fine-tuning result exists. Tables in a group share columns but hold different data |
| Drop tables with blank or duplicate headers, or over 50% empty cells | Us | Such tables make poor training questions. Rejection counts are in the manifest |
| Contamination filter compares column headers only, at Jaccard 0.9 against all four test sets | Us | GitTables and the Wikipedia benchmarks come from different sources, so headers are a cheap check. It cannot catch the same data under different headers |
| GitTables licence recorded as CC0 | Zenodo | The Zenodo record lists `cc-zero`, which corrects the Blueprint's "CC BY 4.0" |

**Can these tasks run on GitTables?** Expected yes, but not yet run. The prompts need only a table with 5 or more columns. All 96 zips hold 1,018,649 tables, and 89,279 pass every filter except de-duplication. Keeping one table per near-identical header set leaves only 5,036, so we keep up to 5, which gives 10,378 (10,053 after the case-insensitive header filter added later). One risk is topic balance: `id_tables` stays near 45% at every limit tested, so the imbalance comes from GitTables and our filters, not this limit. Another risk is that GitTables is more numeric and less Wikipedia-like than the benchmarks, which could reduce the gain from fine-tuning.

## Study settings

| Decision | Source | Why |
|---|---|---|
| Arms: vanilla, no validation, Table-Specialist | Paper | Section 6.1 |
| 3 seeds | Us | Needed to separate an effect from noise |
| 3,000 tables per round | Paper | Given only as an example |
| Each round's training starts from vanilla weights | Paper, partly | Section 6.5 and Fig. 14 find vanilla "consistently better" than continuing. The paper does not say which its main results use. This corrects an earlier note citing Fig. 10 |
| Each round trains only on that round's validated pairs | Paper | Algorithm 1 resets the training sets at the start of every round |
| Each round's generator is the model from the round before; vanilla in round 1 | Paper | Algorithm 1 |
| Train the SQL and Scala models in lockstep | Paper allows it, us (choice) | Section 5 allows the second model to be vanilla or fine-tuned in lockstep, and Algorithm 1 fine-tunes both. Adds about 1.3 hours per round on GPU 1 |
| Vanilla model writes the questions in every round | Us | The paper does not say. The repo's question files are all named for Llama, in fresh table slices per round |
| Use new tables each round within a seed | Repo | The repo slices its tables into 1st, 2nd and 3rd 3K files. 3 rounds use 9,000 of our 10,053 tables |
| FT-no-validation arm runs the same loop without the Validate step | Paper, partly | Section 6.1 removes only validation; it does not state that the rest is unchanged |
| 10 row samples at 50%, random states 0-9; all must agree | Repo, paper | Repo's `prepare_exec.py`; Algorithm 3 |
| Our own SQL-versus-Scala comparison | Us | The repo's validation script runs both languages but never compares them |
| Compare results ignoring row order and column names, with numeric tolerance 0.000001 | Us | Algorithm 3 says only "r != r'". Spark gives no order without a sort, and the two languages name computed columns differently. The tolerance is unpublished |
| Load each SQL sample by writing it to CSV and reading it back with type inference | Repo | The repo's `execute.py` does this. Our first version kept every column as text, so `MAX("Weight")` and `"n" < 50` compared as text. Found while inspecting pairs |
| Ignore rows whose cells are all empty or null, on both sides, when comparing and when deciding a pair is always empty | Us | Spark's CSV writer writes no line for a one-column null row, so an aggregate over no rows gives SQLite one NULL row and Spark a header-only file. Without this, identical queries were rejected. Cost: a Scala program that drops all-empty rows while its SQL keeps them is not detected |
| Treat True and False as equal to 1 and 0 | Us | Type inference stores booleans as numbers in SQLite, while Spark reads the same CSV as text |
| Read Spark's CSV with `\\` as escape when the file contains `\\"`; unreadable output counts as `scala_failed` | Us | Spark escapes quotes this way; pandas otherwise misreads the value or stops. One real file stopped with `ParserError` |
| Reject a pair that returns no rows on all 10 samples | Us | It has shown no agreement about any data. The paper is silent |
| Run SQL first, and stop Scala at the first disagreement | Us | Saves time and cannot change a verdict, because every sample must agree |
| Questions use the repo's prompt and requirement sampler, seeded per study seed; a table gets a new draw if its draw is empty | Repo, us (seeding) | Read directly from the repo's script, with its rates: predicates 70%, other requirements 33% each |
| Temperature 0.7 for questions; up to 128 output tokens for questions and SQL, 256 for Scala | Us | Unpublished. Scala programs are longer: most of the 9 cut-off cases at 256 in round 1 are real truncation, and 160 of 291 round 1 Scala responses reach the limit because vanilla keeps writing after its code block. Round 2 adapters stop by themselves (9 of 290 reached it), so a higher limit would mostly buy unused text |
| Serve each adapter under a name derived from a hash of its weights | Us | The response cache keys on the model name. With the fixed names `sql` and `scala`, the no-validation arm would have reused the Table-Specialist arm's cached answers for the same seed, since both arms share tables and questions |
| After each round, score the new SQL adapter on all 4 test sets with zero-shot and few-shot prompts, 128 tokens | Us | Matches the vanilla baselines exactly, so that scores are comparable. Zero-shot is the repo's default; few-shot is reported beside it |
| Batch size 1% of the data | Paper | Stated |
| Rank 16, alpha 32, dropout 0.05, all linear layers, LR 0.0002 cosine, 3 epochs | Us | The paper's only LR figure is an OpenAI multiplier of 0.5 |

## Scoring

| Decision | Source | Why |
|---|---|---|
| Report repo-faithful and corrected scores | Us | Faithful is comparable with the paper; corrected fixes known defects |
| Corrected score fixes three things in `are_lists_equal`, and does not lowercase gold SQL | Us | Early return on a close cell; division error when gold is 0; row order without ORDER BY. Lowercasing breaks non-ASCII columns such as `"Álbum"` |
| Run the repo's evaluator from source with listed patches, failing if a patch stops applying | Us | It cannot run as shipped: pandas 3 removed its APIs and its paths are placeholders |
| Keep the third-party clone complete and unedited, as a submodule | Us | `main_evaluate.py` imports every evaluator, and an untouched clone shows the code is the authors' |

## Experiment tracking

| Decision | Source | Why |
|---|---|---|
| Local MLflow (`mlflow-skinny`) instead of W&B, with telemetry off | Us | It needs no account and nothing leaves the machine. W&B needs an account |
| MLflow's file store, using its `MLFLOW_ALLOW_FILE_STORE` setting | Us | Its database backend needs more packages. MLflow says the file store is in maintenance mode but still supports it |
| Run IDs that name the stage, then what ran, checked before any GPU time is used | Us | Keeps runs findable. The scheme is in `ts/tracking.py` |

## Safety and machine

| Decision | Source | Why |
|---|---|---|
| Generated Scala and Pandas run only in the Docker sandbox | Repo | The repo requires it, with no host fallback |
| Run one arm per GPU: Table-Specialist on one, no-validation on the other | Us | User allowed both GPUs. Serving and training never overlap within a run, so one GPU can do both. Only the Table-Specialist arm uses Spark, so the arms do not compete for CPU |
| Leave blank cells as the repo loads them: empty text in SQLite, null in Spark | Repo | Both loaders are the repo's. `COUNT` and non-empty filters can therefore differ on columns with blanks, and such a pair is rejected (`gt_05550`: SQL counts 1, Spark counts 0). Changing it would depart from the repo, and the paper is silent |
| Run the two arms at the same time | Us | Saves about 47 hours. Cost: the arms write their own round 1 questions, since neither finds the other's in the cache, so the same seed need not give identical questions in both arms |
| Raise the sandbox's open-file limit from 256 to 4,096 | Us | At 256, Spark 3.5.7 cannot start: 0 of 32 programs ran, failing with `tzdb.dat (Too many open files)`. It loads through a checked patch, leaving the clone unedited. No network, read-only filesystem, non-root user and the memory limit are unchanged |
| Spark base image `apache/spark:3.5.7`, pinned by digest | Us | The repo asks for a Spark 3.5.x image by digest. Digest is in `configs/sandbox.env` |
| Model SQL runs read-only, SELECT only, with a 5-second limit | Us | SQLite runs on the host |
| GPU 0 serves, GPU 1 trains | Us | GPU 1 drives the display and has about 1.75 GB less free memory |
| `VLLM_USE_FLASHINFER_SAMPLER=0`; no `expandable_segments` | Us | Both found necessary on this machine under WSL2 |

## Bug decisions during the core study

| Decision | Why |
|---|---|
| JSONL readers split rows only on `\n`, not with `str.splitlines` | `splitlines` also breaks on U+2028 and similar characters. Training files keep these raw, so a row broke and training stopped. `read_journal` would have dropped such a row silently and redone it |
| Keep the U+2028 characters in the training text | They come from the table text. Removing them would change the data for no benefit, and the tokenizer accepts them |
| Fix files in place while Table-Specialist runs | Each training step starts a new process, so it picks up the fix; the running loop keeps its old reader, which only reads files without raw U+2028. Scores do not change, as rows that load are unaffected |
| Restart no-validation at round 2 training, not from the start | Rounds and steps resume from saved files. Round 1 used no affected row: its files hold no U+2028 |

## Crash on 2026-11-05 and recovery

| Decision | Why |
|---|---|
| Long jobs run in their own detached tmux session, logged through `tee`, one job per GPU | The user's rule after the crash. This replaces `nohup setsid` under the Claude session |
| The crash's cause is recorded as unknown | The jobs were already in their own sessions and still died. The journal has no entries from 06:04 to 09:29 UTC, so it cannot show or rule out an out-of-memory kill |
| A finished round is left untouched by a re-run | A re-run used to rewrite `summary.json` with zero timings and add a duplicate MLflow run. `no_validation` round 1 already has both; its real timings remain in the first `ROUND` line of its log |
| Training logs are appended, not replaced | A resume would otherwise erase the interrupted run's log |
| Checkpoint every 1/12 of a run, or about 5 minutes of training if shorter; a resumed run also uses the new interval | The old rule meant up to 14.6 minutes lost on the largest adapters. Checkpoints do not change the training itself |
| Copy `checkpoint-275` and `checkpoint-72` to `runs/_crash_backup_2026-11-05/` before resuming | The trainer removes checkpoints as it goes, and the user asked that existing ones be kept |
| Resume the two arms in parallel, each stopping at the stage the user named | The user's choice; about 1.9 h against about 3.4 h one at a time |
| No git commits | The user's decision, 2026-11-05 |
| Claude is notified when a tmux job ends and starts the next job itself | The user's decision, 2026-11-05: starting each job by hand is inefficient. The job stays in tmux; only a blocking `tmux wait-for` runs as a background task, with no polling. If the Claude session dies, jobs continue but the next one is not started until the session is back |
| In the rest of the study, `table_specialist` validation may overlap `no_validation` training on the other GPU | The user's decision, 2026-11-05. Validation uses the CPU, and the overlap cost about 7-16% in training speed. The crash window is missing from the journal, so a part in the crash cannot be ruled out; separating them would add many hours |

## Extension 2: learning from rejected programs (decided 2026-11-07)

Not in the paper. The user chose Extension 2 of the mid-sem deck as the first novelty, ahead of Extension 1 (validator precision and recall) and Extension 3 (small specialists), and asked that both GPUs be used wherever possible. Code: `ts/ext2.py`, `ts/train/dpo.py`, settings under `extension2` in `configs/base.yaml`, tests in `tests/test_ext2.py`.

| Decision | Reason |
|---|---|
| Pilot on round 1 only, 3 seeds, on the same questions and tables as `table_specialist` round 1 | Round 1 is where TS lost to NV 7 to 17, so a gain is easy to see; reusing the questions makes the comparison paired. Rounds 2-3 follow only if round 1 improves |
| 4 candidates per question: TS's greedy pair (verdict reused) and 3 pairs sampled from vanilla at temperature 0.7 with request seeds `seed*100 + j` | DPO needs a better and a worse answer to the same question; TS has one program per question. 0.7 is the config's question temperature |
| Chosen = the first candidate, in order, whose SQL and Scala agree on all 10 samples (paper Algorithm 3, unchanged) | The validator stays the paper's |
| Rejected = a candidate whose SQL rows differ from the chosen SQL's on some sample, fail, or are always empty. Never judged by its own Scala | A correct SQL program whose Scala failed would otherwise become a negative. Candidates with the same rows as the chosen one are `same_result` and not used |
| Spark runs only until a question has a chosen program; later candidates are labelled by SQL results alone | Same labels, far fewer Scala runs. SQL runs in SQLite in milliseconds |
| Two models, SQL only, both from vanilla: `bestof4` (SFT on one chosen program per question) and `dpo` (`bestof4` then DPO on the pairs) | `bestof4` against TS isolates the effect of extra sampling; `dpo` against `bestof4` isolates learning from rejected programs. Only SQL is scored, so no Scala adapter in the pilot |
| DPO reference = the SFT adapter (TRL copies it as a frozen `ref` adapter); reference log-probabilities computed before training | The usual DPO setup; precomputing saves the memory a second forward pass would need |
| DPO plus an SFT term on the chosen program, weights 1 and 1 (RPO); beta 0.1; LoRA learning rate 2e-5, cosine, 10% warm-up; 2 epochs; 16 pairs per step | All ours. Plain DPO can lower the chosen answer's likelihood; the SFT term guards against it |
| Only one seed validates at a time (`runs/.spark_validation.lock`); each GPU runs its own seeds | Validation uses 36 of 40 CPUs. GPU 0 runs seeds 1 and 3, GPU 1 runs seed 2: while one seed validates, the other GPU generates, trains or evaluates |

### DPO tuning on dev sets (decided 2026-11-08)

The first DPO settings lowered scores (see `docs/RESULTS.md`). The user approved tuning DPO on dev sets while `bestof4` runs.

| Decision | Reason |
|---|---|
| Tune on dev sets only, never on the test sets | Choosing settings by test scores would bias the reported numbers |
| Dev sets (`ts/bench/dev.py`): WikiSQL dev, 300 cases; Spider dev single-table, 575; WikiTQ `random-split-1-dev`, 300. Zero-shot only | Each comes from a split no test set uses. BIRD has no unused split downloaded (its test set is BIRD dev), so BIRD is left out. Zero-shot is the main evaluation prompt |
| Builders gained a `split` argument; the evaluator a `--split dev` option with its own folder (`data/bench/NL2SQL/benchmark/dev/`) and ID lists (`results/ids/dev/`) | Defaults are unchanged, so test sets are built and read exactly as before; the test folders were not touched |
| Variants and the selection rule written in `configs/base.yaml` before any dev result | So the choice cannot be fitted to the results afterwards |
| Tune on seed 1's pairs and `bestof4` adapter; reference points TS, `bestof4` and the first DPO, all scored on dev the same way | Seed 1's pairs exist, so no new validation is needed |
| Selection: highest mean zero-shot dev accuracy. The winner runs on all 3 seeds and the test sets only if it beats `bestof4` on dev | Otherwise DPO is reported as not helping, with the dev evidence |
| v4 and v5 run on GPU 0 after v1-v3 (09:31 IST), not on GPU 1 after seed 3 as first planned | GPU 0 was free about 1.5 h before GPU 1; same model, data and settings, so the change affects timing only |

### Extension 3: repair rejected pairs (decided 2026-11-08)

The user chose this after the Extension 2 results; bestof4 rounds 2-3 are on hold. Code `ts/repair.py`, settings `configs/base.yaml` `repair`.

| Decision | Reason |
|---|---|
| One repair attempt per rejected pair, then Algorithm 3 again, unchanged | The paper discards rejected pairs with no repair; 259 of 300 pairs (86%) are rejected in the 300-table check |
| A program's feedback comes only from its own run: the SQLite or Spark error, "returns no rows", or its own first 5 rows on the sample where the pair was rejected. It never sees the other language's code or output | So that agreement after repair is still two independent programs agreeing, and repair cannot copy the other side's mistake |
| sql_failed and always_empty: repair the SQL. scala_failed: repair the Scala. disagree: repair both separately and try (new SQL, old Scala), (old SQL, new Scala), (new, new), keeping the first that passes. no_code: not repaired | Repair only what the failure points at; Scala-only repair leaves the SQL, which is the training label, untouched |
| Prompt: the repo's generation prompt, the failed code block, its feedback, then the repo's correction phrase "Upon review, code is incorrect and should be rewritten as follows:" from `nss_task_descriptor.get_output` | The phrase is the repo's own format for a corrected answer; the base model has no chat template |
| Generator: vanilla, as round 1 | The same model that wrote the rejected pair |
| Greedy first; if the reply is empty or identical to the failed code, up to 3 samples at temperature 0.7 (request seeds `seed*100+1..3`) | In the first smoke test (10 pairs) 6 of 10 greedy repairs copied the failed code exactly. This was decided on those 10 pairs, which are part of the pilot set; the pilot also reports the result without them |
| Pilot: seed 1's 300-table set (`runs/train_tablespecialist_base_seed1_test300/round1`), output `runs/repair_pilot_seed1_test300/round1` | Existing data, about 1-2 h |
| Go / no-go, fixed before the pilot: kept pairs +30% or more (41 → 54 or more), and 80% or more of hand-checked recovered pairs correct. The first-pass kept pairs scored about 78% (39 of 50) in the earlier hand check | Below this, repair adds too little data or too much noise to be worth full runs |
| Base pilot no-go (+17.1%). Second pilot, approved by the user: Llama-3.1-8B-Instruct (revision 0e9e39f, already downloaded) writes the repairs through its chat template; same rejected pairs, same diagnoses (`repair_plans.jsonl` reused, read only), same retries, Algorithm 3 and go / no-go rule. Prompt: the repo prompt, failed code and feedback in one user message, asking for the corrected code only. Reply limits 256 tokens (SQL) and 512 (Scala) | Base Llama copied the failed code back in 85 of 169 Scala and 42 of 158 SQL repairs; an instruction-tuned model of the same family and size should follow a repair request. A second model then helps build the data, so results must say the data is no longer from the base model alone |

| Full run after the Instruct pilot's go (agreed plan): `ts/repair_ts.py`. Per seed, every pair TS round 1 rejected is repaired by Instruct; a SQL adapter is trained from vanilla on TS round 1's kept pairs plus the recovered pairs (the recovered pair's SQL is the one that passed), with the core study's SFT settings, and scored on the 4 test sets with both prompts. Run IDs `train_repairts_base_seed{1,2,3}` | Same questions, tables and training settings as TS round 1, so the only difference is the extra repaired pairs. SQL adapter only, as for bestof4. TS's run directories are only read |
| Lanes: GPU 0 runs seeds 1 then 3, GPU 1 runs seed 2. Repair and validation take the shared validation lock (36 CPUs), so one seed validates at a time while the other trains or evaluates | Both GPUs used; the CPUs are the limit |
| Rounds 2-3 (user approved 2026-11-08, 23:50 IST, after round 1 beat TS round 1 on all 3 seeds): `ts/repair_loop.py`. Round 1's directory gains a Scala adapter trained from vanilla on the Scala of the same kept + recovered pairs (`round1/train_scala.jsonl`, new files only). Each later round follows Algorithm 1 as `ts/loop.py` does (vanilla questions on the same tables as TS's round, the previous round's SQL and Scala adapters write code, Algorithm 3), then Instruct repairs the rejected pairs (output in `round{r}/repair/`), and both adapters train from vanilla on kept + recovered pairs; the SQL adapter is scored on all 8 test cells | Algorithm 1 needs both adapters to generate in round 2; repair is added to every round so the arm differs from TS only by repair |
| Rounds 2-3 scheduling: all 3 seeds run at once, one process each. Every GPU step (generation, repair generation, training, evaluation) takes whichever of the 2 GPUs is free (`gpu_slot`, lock files `runs/.gpu{0,1}.lock`, port 8000 + GPU); Spark validation keeps the single validation lock. The SQL and Scala adapters train in parallel when both GPUs are free, and evaluation starts as soon as the SQL adapter is done | The CPUs (validation) are the bottleneck, about 4-5 h per seed-round; with 3 seeds queued, the validation slot is never idle and each GPU still runs one job at a time |

### Plan A and B: match the paper, then improve (decided 2026-11-10)

The user approved `docs/PLAN_NEXT.md` ("Go ahead"), with my recommended answers to its 3 questions: Instruct is treated as the paper's model for matching (base stays the faithful reproduction), improvements use established methods only, and A3 runs.

| Decision | Source | Why |
|---|---|---|
| A1: Instruct vanilla on all 4 test sets, chat with the repo's system message, 256 tokens, zero-shot and few-shot (`ts/a1_instruct_gate.py`, run IDs `gate_instruct_*`) | Repo (chat call, system message) | The repo's only query path is chat with that system message; earlier Instruct scores were within 0.011 of the paper on WikiSQL and WikiTQ |
| A2: `TS_PROFILE=instruct` runs the core loop on Instruct: generation, training and evaluation all through the chat template with the repo system message; training examples in TRL's conversational prompt-completion form; limits 256 (questions, SQL) and 512 (Scala) tokens; run IDs `train_<arm>_instruct_seed<s>`. Unset, the loop is unchanged | Repo (chat call); limits ours | Training must see the same template as serving. Chat answers carry prose, so the base limits would cut answers |
| B1 + B2: execution-based self-consistency (`ts/eval/self_consistency.py`): greedy answer plus 8 samples; failed or empty results dropped; the first answer of the largest group of identical results returned; greedy kept if nothing is usable; the gold answer is never used for the choice | Wang et al. 2022; OmniSQL; Wang et al. 2018 | Established test-time methods; no training |
| B1 tuning: temperature 0.5, 0.7 or 1.0 on the dev sets, zero-shot, TS round 3 seed 1 adapter; highest dev mean wins, and goes to the test sets only if above the greedy dev mean (`configs/base.yaml` `self_consistency`, fixed before any dev result) | Us | Same protocol as the DPO tuning |
| A2-A3 and B1 share both GPUs through the GPU pool (`ts.loop --gpu-pool`, `ts.sc_run`): each GPU step takes whichever GPU is free; Table-Specialist validation takes the shared validation lock | Five jobs (3 TS seeds, NV seed 1, B1) on 2 GPUs; validation stays one at a time on the 36 Spark workers |
| B1 result (2026-11-11): self-consistency is reported **beside** greedy, not in place of it. The paper's numbers are greedy; the vote needs 9 generations per case and is a test-time method from the literature, not part of Table-Specialist | Keeps the reproduction comparable with the paper while showing the established improvement |
| B1's greedy BIRD score is 3 cases below `run_baseline`'s. Kept as measured and flagged, not corrected. Gains are measured within each run | A within-run comparison is fair either way; correcting it would mean guessing the cause before checking it |
| BIRD cause found (2026-11-12): 20k-50k-row results hit the 5 s timeout under 16 threads. BIRD is now scored on one thread; the 14 base BIRD cells are re-scored under new IDs (`final_sct10seq_*`), and the old files are kept | Matches `run_baseline`, which scores on one thread; new IDs leave the existing run directories untouched |
| B1 on Instruct runs each Instruct TS seed as soon as its round 3 has finished, sharing the GPU pool with A2/A3 | Uses a GPU freed by seed 1 without waiting for the other seeds |
| Reversed 2026-11-12, 08:26 IST: B1 jobs wait until A2/A3 have finished | The GPU pool is first come, first served. B1 held GPU 0 for 2.5 h while A2 seed 2 waited to train. Matching the paper (A2) comes first; B1 loses nothing (finished cells skipped, generations cached) |
| Repair-TS on Instruct (plan step 4, started 2026-11-12) runs exactly as on base: round 1 repairs that seed's Instruct TS round 1; the Instruct model is both generator and repairer; prompts, rules and training are unchanged; chat format throughout (`TS_PROFILE=instruct`) | A2 matched the paper (zero-shot mean of 4: 0.6113 against 0.5983). The approved plan runs repair-TS on Instruct if it does. On base, the repairer was already the Instruct model |
| B1 (self-consistency, T 1.0, 8 samples, same settings) also runs on Instruct repair-TS round 3, one seed at a time as each finishes (`sc_run --only instruct_repairts_seed{s}`, started 2026-11-14, 05:53 IST for seed 1). One job at a time on the GPU pool, so one GPU stays free for repair-TS seeds 2-3 | On base, repair-TS r3 + vote is the best result (0.6015 mean of 8). The final table needs the same row on Instruct. Both GPUs were idle while seeds 2-3 wait on CPU validation |
| Phase 8 (approved 2026-11-14): the next task is NL-to-Scala, on **Llama-3.1-8B-Instruct only** (vanilla, TS and repair-TS, seeds 1-3, rounds 1-3, both prompts: 19 models). Base is dropped for now; its Scala adapters exist and can be scored later | NL-to-Scala needs no new data or training: every round already trained `adapter_scala` on the validated (and, for repair-TS, repaired) pairs. Instruct is the model that reproduced the paper's SQL numbers; dropping base halves the CPU time (about 63 h instead of 125 h) |
| NL-to-Scala test scoring follows the repo's `nss_evaluator.py`: zero-shot prompt from its NSS generator (the format the adapters were trained on), few-shot from `nl2sparkscala_generation_v6.py`; the last ```scala block; the code must read `data.csv` and write `result`; Spark runs on the **full** table (the repo's `csv_path`; for BIRD the prompt shows 10 rows); gold is the gold SQL run on the case database; repo-faithful = the repo's exact, order-sensitive list equality after `to_numeric` and `fillna("None")`; corrected = our order-insensitive SQL rule (`ts.eval.compare`) on the same results | Same structure as the SQL evaluation, so the Scala numbers compare with the paper's in the same way. The repo's comparison is order-sensitive although Spark gives no row order without a sort, so the corrected score is reported beside it, as for SQL |
| Scala test runs are deduplicated: identical (code, table) pairs run once in Spark and the result is reused across models | Seeds and rounds often write the same code for simple questions. A verdict depends only on the code and the table, so this changes no score |
| Phases kept apart so the jobs do not slow each other: all Scala code is generated first on the GPUs, then scored on the CPUs (36 Spark workers, under the validation lock); error detection uses the GPUs while Spark scores. The Spark throughput is measured with and without error detection running; if it drops by more than about 10%, Spark goes to 32 workers or the jobs run one after the other | A side job holding a GPU once delayed a main run by 2.5 h. Error detection needs no Spark, so the two use different hardware |
| Error detection is scored with the repo's `ed_evaluator.py` metric (precision averaged over columns with a prediction, recall over columns with errors, F1 from the averages; gold = AutoTest `ground_truth`), zero-shot with the repo's prompt; AutoTest's own rule (predictions of `ground_truth_debatable` values count neither way) is reported beside it | The repo's evaluator has no notion of debatable values, and the paper does not say how they were treated. Few-shot is not run: the repo's ED test uses the zero-shot prompt |
| Error-detection answers may use up to 512 tokens (was 256) | The repo's query script sets no limit. In the smoke test (2026-11-14, 17:46 IST), Instruct explains before its JSON, and 2 of 40 answers were cut at 256 tokens before the JSON, counting as no prediction. 512 is the Instruct limit used for Scala |
| First NL-to-Scala smoke test (17:42 IST) is void: its tmux command did not source `configs/sandbox.env`, so every program failed with "Set TABLE_SPECIALIST_SPARK_IMAGE". Fixes: the scorer calls the sandbox's `ensure_sandbox` before running anything; the execution cache moved to `runs/nlscala_exec_cache/` (the old `runs/scala_exec_cache/`, 132 records of that error only, is left in place and never read); the rerun uses tag `r2` (`smoke_nlscalar2_*`), so the void `smoke_nlscala_*` results are not overwritten. Every Phase 8 tmux command sources `configs/sandbox.env` and `configs/runtime.env`, as `scripts/run_study.sh` does | Found by the smoke test: 132 of 132 programs failed in 0.02 s each |
| Error-detection TS (`ts/ed_loop.py`): per round the seed's next 3,000 corpus tables (as for SQL); typo and semantic errors written by **vanilla** (temperature 0.7, as for questions); the **validator is the previous round's error-detection adapter** (vanilla in round 1, greedy), as Algorithm 1 fine-tunes M_C each round; each round trains from vanilla on its own kept columns only; only columns whose repo test prompt fits the length limit are used | The paper and repo say which prompts and checks to use but not which model writes the errors or how rounds are chained for this task; these choices mirror the SQL study (configs/base.yaml: question_model vanilla, restart from vanilla, new data only) |
| Machine stall 2026-11-14 19:37 IST to 2026-11-15 03:03 IST: the 36 Spark programs cached as "Sandbox operation timed out" after 26,891-26,955 s are treated as missing and re-run. Any cache record that timed out after more than 600 s counts as stalled. The re-run is stored beside it as `<key>.rerun.json`, and the original is never overwritten. The running `nlscala_score1` was stopped before it wrote any cell (it held those records in memory) and relaunched as `nlscala_score2`; all finished programs stayed cached. The failed error-detection smoke test is rerun under a new tag (`smoke2`), so `train_edtssmoke_instruct_seed1` is left as it is | The Scala limit is 120 s, so a 7.5 h run time can only come from the stall, not from the code; these 36 were the only timeouts in the cache. Keeping the originals and using a new tag follows the rule that no run directory is deleted, moved or overwritten |
| Speed fix in `ts/eval/compare.py` (2026-11-15): the order-insensitive comparison computes the two row-count differences once per comparison. Before, it rebuilt them for every row, which is quadratic: about 10 min for the 49,729-row BIRD answer and about 43 min per TS BIRD cell. Running jobs keep the old code they loaded; the rounds 1-2 Scala scoring uses the new code. New test `test_large_unordered_results_compare_quickly`; suite 191 passed, 1 skipped | A pure hoisting of an unchanged value, so every verdict is identical. Checked read-only: all 51 written Scala cells (45,930 cases) rescored with the new code give 0 differences in repo verdict, corrected verdict and outcome, in 149 s in total. Published NL-to-SQL corrected scores are unaffected for the same reason |
| Same-data control for repair-TS (2026-11-13): extra pairs are other seeds' TS round 1 kept pairs, not newly generated ones | Same generator (vanilla), same Algorithm 3 check, only the amount of data differs. Needs no Spark validation, so it runs beside repair-TS without slowing it. The user asked that repair-TS finish undisturbed and that other useful work run alongside |
| Significance: paired bootstrap over test cases on the mean of 8 (equal cell weights), with per-seed differences listed | Matches the headline statistic; the case bootstrap cannot cover seed variation, so the seeds are shown separately |
| Repair-ED (2026-11-15, `ts/repair_ed.py`, not in the paper): each error the validator rejects gets one more attempt from vanilla (temperature 0.7, seed as TS), then Algorithm 2 again, unchanged, with the same validator. The repair prompt is the repo's generation prompt, the model's own answer, what the validator did on the 5 shuffles (how many found the error, how many found nothing, up to 3 other values it named, each cut to 60 characters), and our phrase modelled on the repo's code-correction phrase: "Upon review, the error is not obvious enough and should be rewritten as follows:". A retry that is unparsable, empty, already in the column or the same error is dropped. Rejected clean columns are not retried. 3 seeds x 3 rounds; the validator in rounds 2-3 is repair-ED's own previous adapter | The ED analogue of repair-TS: feedback from the failed check, the check itself unchanged. Clean columns have no error to rewrite |
| Repair-ED reuses TS's generated errors (TS `errors.jsonl` of the same seed and round, read only) and, in round 1, TS's `validated.jsonl` (same vanilla validator) | Identical by construction (same model, prompts, request seed and tables), so repair-ED differs from TS only by the retries and the later validators; saves about 40 min per seed |
| Best-of-N control for repair-ED (`--variant resample`, round 1, 3 seeds): each rejected error gets one more sample from the same generation prompt, no feedback (request seed + 1000, so it is not the cached first sample). Same-data control (`--variant samedata`, round 1, 3 seeds): TS round 1's kept columns + as many extra kept typo and semantic errors as repair-ED round 1 recovered, matched by kind, sampled with Random(seed) from donor seed (s % 3 + 1) TS round 1, excluding all 3,000 tables of this seed's round 1 | Resample separates the feedback from simply having a second attempt; same-data separates where the extra examples come from from how many there are. Both as for repair-TS (`ts/data_control.py`) |

## Phase 9: NL-to-R (approved by the user 2026-11-16, 08:40 IST: "do what you can for NL to R without hampering the current running experiments of error detection")

| Decision | Why |
|---|---|
| Same scope as NL-to-Scala: Llama-3.1-8B-Instruct; vanilla, TS and repair-TS, seeds 1-3, rounds 1-3, both prompts; the 4 SQL test sets | Directly comparable with the NL-to-SQL and NL-to-Scala results |
| Dual task NL-to-SQL: each round, SQL and R written for the same question must agree by execution on 10 row samples (Algorithm 3 with R in place of Scala; `ts/validate/r_agree.py`); two adapters per round (SQL and R) from vanilla on the kept pairs | The repo has no NL-to-R validation script (`Validation/NL2DSL/` holds only NL2SQL_SparkScala). The paper describes NL-to-R as validated against a second language; SQL is the one with gold test answers and our checked SQL runner. Same samples, comparison and tolerance as the main study |
| Questions are TS's own (`runs/train_tablespecialist_instruct_seed<s>/round<k>/questions.jsonl`, read only), and round 1's SQL is TS round 1's `sql.jsonl` | Identical by construction (vanilla, same prompt, request seed and tables; checked: all 9 seed-rounds cover exactly the round's 3,000 tables). Saves about 1 h of GPU per seed-round. Rounds 2-3 SQL and all R are generated by this run's own adapters |
| Repair-TS for R: as Instruct repair-TS for Scala (`ts/repair.py` rules): feedback from what each program did on the rejecting sample (SQL error, no rows, R's `Error` lines, or both outputs), the Instruct chat repair prompt, greedy then up to 3 retries at 0.7, candidates (new SQL, old R), (old SQL, new R), (new, new), Algorithm 3 again unchanged. Round 1 starts from TS-R round 1's validated pairs | The same method as the Scala experiment, so the two can be compared |
| R prompts and formats from the repo: generation `Generation/NL2DSL/NL2R/nl2r_generation_v6.py` (few-shot; read from its syntax tree), training and zero-shot test format `Finetune/data_generator/nl_to_r` (input `(table, question)`), completion ```` ```r ... ``` ```` | Byte-for-byte repo prompts, as for SQL and Scala |
| R test scoring follows `Evaluete/evaluate/nr_evaluator.py` (`ts/eval/r_eval.py`): the FIRST ```` ```r ```` block, no format check, full table, `result.csv` read with default NA handling, to_numeric, fillna("None"), exact list equality; corrected score order-insensitive unless ORDER BY. Gold shared with NL-to-Scala | The repo's own R rules (they differ from its Scala rules: first block not last, no format check, no numeric tolerance) |
| In validation, R's unquoted NA is read as an empty cell | SQL NULL and Spark null are empty cells in `exec_agree`; R writes them as NA |
| R sandbox: our `sandbox/Dockerfile.r` = the repo's (r-base by digest, user 65534) + Debian's `r-cran-dplyr`. Base `r-base:4.4.2` (digest fe9b2952...); installing dplyr upgraded R to 4.6.1 (dplyr 1.2.1); image `sha256:b8be8758...` in `configs/sandbox.env` (previous file copied to `backups/sandbox.env.before_r_20261116`) | The repo's Dockerfile has no R packages and says to add the benchmark's; every repo NL-to-R script starts with `library(dplyr)`. Network is off inside the sandbox, as before |
| Max tokens for R: 512 (as Scala) | An R script is about as long as a Scala one, and chat replies add text |
| GPUs: NL-to-R takes only the GPUs named in `TS_R_GPUS`; until error detection finishes it uses only GPU 0, after the error-detection controls have left it | The user asked that the error-detection runs are not slowed. One job per GPU is kept by the GPU locks |

## Open questions

Contacting the authors is not possible (decided by the user, 2026-11-04), and the draft email will not be sent. These questions stay unresolved unless the paper or repo settles them, and the results must say so.

1. Which prompt and API produced the paper's vanilla scores? Base scores 0.326 (zero-shot) and 0.685 (few-shot) on WikiSQL against the paper's 0.525. Unexplained.
2. Which checkpoint did the paper use? The paper never says base or Instruct; the repo's README link is our evidence.
3. What removes 7 BIRD cases?
4. Were WikiSQL cells lowercased in the paper's prompts?
5. Which exact test cases did the paper use? No case ID lists are released.

### 2026-11-16 15:05 IST: R repair feedback keeps the whole error block

The NL-to-R smoke test showed repair feedback such as "Error in `filter()`:" with no cause: dplyr (rlang) prints the cause on the following lines ("Caused by error: ! object 'Access' not found"), and `r_messages` only kept indented continuation lines. Now a message is the Error line plus every following line up to `Backtrace:`, `Execution halted`, a blank line or the next error, capped at 300 characters. This is the R equivalent of what the Scala repair feedback shows (the compiler error); the smoke's repair round 1 (0 of 26 recovered) ran with the old messages and is not used. SQL validation errors such as OverflowError (a Python int too large for SQLite) stay a separate rejection reason, as in the SQL-Scala loop (1 of 30 in the smoke).

### 2026-11-16 20:12 IST: NL-to-R holds its GPU through validation

`ts.repair_loop.gpu_slot` takes the first free GPU, trying GPU 0 first. When TS-R released GPU 0 for its CPU-only validation, the running repair-ED took GPU 0 for its next step and kept it for about 2 h, so TS-R waited and GPU 1 was idle. Results are unaffected (each step is self-contained), only time was lost. TS-R now runs with `TS_R_HOLD_GPU=1`; repair-ED's later runs take `TS_ED_GPUS`. The running repair-ED process keeps its old behaviour, which is now harmless because TS-R never leaves GPU 0 free for more than a second.
