# Tasks

Last updated: 2026-11-15, 18:20 IST.

A box is ticked only when a file, log or test result shows it is done. "Running" means started with no result yet.

**Model:** Llama-3.1-8B base is the faithful model. Instruct is a future experiment.

**Now (2026-11-11, 15:40 IST):** Phases 1-5 are finished; repair-TS is finished for 3 seeds × 3 rounds. Phase 6b (docs/PLAN_NEXT.md) is under way: A1 passed; B1 self-consistency on the test sets is finished (repair-TS + vote, zero-shot mean of 4, 0.6051 against the paper's TS 0.5983). Running in tmux, sharing both GPUs: A2 Instruct TS seeds 1-3 (`its_seed{1,2,3}`) and A3 Instruct NV seed 1 (`inv_seed1`). Round 1 of Instruct TS is already above the paper's TS: 0.6280 and 0.6096 for seeds 1 and 2. Projected finish: Thu 12 Nov, about 10:00-14:00 IST. The base Phase 3 gate is still not passed; the Instruct gate passed on 3 of 4 sets.

## Running

- **2026-11-11, 15:40 IST:** `its_seed1`, `its_seed2`, `its_seed3` and `inv_seed1` are running on the GPU pool (`--gpu-pool`), logs `logs/instruct_ts_seed{s}.log` and `logs/instruct_nv_seed1.log`. `sc_test` finished with exit 0 at 15:35 IST. The waiter has been re-armed; the `session-closed` hook is still set
- Resume after the crash on 2026-11-05 (about 08:15 UTC, 13:45 IST), started 15:45 IST. Each job has its own tmux session and GPU. `study_nv` (GPU 1): finished with exit 0 at 16:51 IST. `rest_nv` (GPU 1), started 16:52 IST: `no_validation` round 3 and seeds 2-3, logged in `logs/study_no_validation_driver.log`. `study_ts` (GPU 0): finished with exit 0 at 17:35 IST, completing seed 1 of `table_specialist`. `rest_ts` (GPU 0), started 17:36 IST: `table_specialist` seeds 2-3, logged in `logs/study_table_specialist_driver.log`; finished with exit 0 at 03:12 IST on 2026-11-07, completing all 3 seeds of `table_specialist`. GPU 0 is idle until the user approves Phase 6. `rest_nv` finished with exit 0 at 19:00 IST on 2026-11-07, completing all 3 seeds of `no_validation` and the core study. Nothing is running; both GPUs are idle. The tmux server has exited, so the `session-closed` hook must be set again before the next job
- Both rest-of-study jobs were started by Claude after a clean exit (user's decision, 2026-11-05). Validation may overlap training on the other GPU (user's decision, 2026-11-05)
- No commits: the user decided on 2026-11-05 not to commit anything. Notification: `tmux wait-for job_closed`, started as a background task, blocks without polling until a tmux session closes (hook `session-closed`). Claude then checks the exit status and starts the next job

## Crash recovery / how to resume

Every long job runs in its own detached tmux session, one job per GPU. tmux survives an editor crash. It does not survive WSL shutting down or a power cut, and restarting after those is manual.

1. Check that nothing is running: `tmux ls`, `pgrep -af 'python.*-m ts\.'`, `nvidia-smi`.
2. Copy any checkpoint you want to keep. The trainer replaces it at its next save and removes the last one when an adapter finishes. Copies from the 2026-11-05 crash are in `runs/_crash_backup_2026-11-05/`.
3. Run the same command again. It skips everything finished and continues the rest, whichever stage stopped:

```bash
# The rest of one arm, all seeds:
tmux new -d -s rest_nv 'cd /home/ts/table-specialist-repro && scripts/run_study.sh no_validation 1 8001 2>&1 | tee -a logs/study_no_validation_driver.log'
tmux new -d -s rest_ts 'cd /home/ts/table-specialist-repro && scripts/run_study.sh table_specialist 0 8000 2>&1 | tee -a logs/study_table_specialist_driver.log'
```

To stop at a chosen seed and round, run `ts.loop` directly and add `--seed` and `--rounds`:

```bash
tmux new -d -s study_nv 'cd /home/ts/table-specialist-repro && set -a && . configs/sandbox.env && . configs/runtime.env && set +a; CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m ts.loop --arm no_validation --seed 1 --rounds 2 --serve-gpu 1 --train-gpu 1 --port 8001 2>&1 | tee -a logs/study_no_validation_seed1.log; echo "exit=${PIPESTATUS[0]} $(date -u)" | tee -a logs/study_no_validation_seed1.log'
```

| Stage | Skipped when | If interrupted |
|---|---|---|
| Whole round | `round<k>/summary.json` exists | Left untouched |
| Generation | `questions.jsonl`, `sql.jsonl` and `scala.jsonl` all exist | Generation runs again; the response cache returns finished responses |
| Validation | `validated.jsonl` exists | Continues from `validated.journal` |
| Training | `adapter_<language>/train_summary.json` exists | Continues from the newest `checkpoint-*`; the log is appended |
| Evaluation | `results/baseline/<run>.json` exists for that test set and prompt | Only the unfinished test set and prompt run again |

A resumed round's summary has `"resumed": true`: its timings, and the `seconds` and `train_loss` of a resumed adapter, cover only the work after the restart. The full loss history is in the training log.

## Phase 1: Environment

- [x] WSL2, Ubuntu 24.04, both GPUs visible
- [x] Python 3.12.3 venv with vLLM, TRL, PEFT, bitsandbytes, sqlglot, pandas
- [x] Llama access and Hugging Face login
- [x] Download Qwen2.5-7B-Instruct
- [x] Download Llama-3.1-8B-Instruct at `0e9e39f`; all 4 checksums match
- [x] Download Llama-3.1-8B base at `d04e592`; all 4 checksums match
- [x] Build the Python sandbox; `tests/test_sandbox.py` passes (26 tests)
- [x] QLoRA and vLLM smoke tests on Instruct
- [x] LoRA smoke test on Instruct: 20.0 GB at about 4,000 tokens
- [x] LoRA smoke test on base: 19.8 GB at about 4,000 tokens
- [x] Serving smoke test on base: prefill 7,417 and decode 262 tokens/s
- [x] Build the Spark sandbox and measure its speed: 48 of 48 programs, 39.7 s each
- [x] Run the repo's sandbox integration tests: 5 passed, 1 skipped (R) with our open-file patch; Spark fails without it
- [x] Try more Spark workers: 36 workers run 2,714 programs per hour (11.1 hours per round)
- [x] Set up local MLflow and a run-ID scheme (`ts/tracking.py`)
- [x] Record the host: Xeon w7-3455, 40 logical CPUs, 102 GB memory, 2 RTX 4500 Ada, 878 GB free
- [x] Watcher for jobs started outside the session (`scripts/watch_until_done.sh`)
- [ ] First git commit (waiting for approval; only the submodule is staged)

## Phase 2: Data

- [x] Download WikiSQL, WikiTQ, Spider, BIRD, AutoTest
- [x] Finish the GitTables download: 96 zips, 1,018,649 tables
- [x] Build WikiSQL: 1,000 cases
- [x] Build Spider: 1,230 cases, 1,198 within the prompt limit
- [x] Build WikiTQ: 1,000 cases
- [x] Build BIRD: 363 cases (paper has 356)
- [x] Record the 7 extra BIRD cases: checked and unexplained; see `docs/DECISIONS.md`
- [x] Count AutoTest: 1,200 columns per file against the paper's 1,126 and 1,081; the paper's filter is unknown
- [x] Write `ts/corpus/build.py`: limits, MD5 check, contamination and duplicate filters, with tests
- [x] Verify all 96 GitTables zips against Zenodo's MD5: no problems
- [x] First corpus build: only 5,036 of the 20,000 target, because keeping one table per near-identical header set removed 84,243 of the 89,279 that passed the other filters
- [x] Measure tables kept per limit: 5,036, 7,430, 8,751, 10,378, 12,640 at limits 1, 2, 3, 5, 10
- [x] Rebuild the corpus with a limit of 5: 10,378 tables; `id_tables` is 45% (`results/corpus_manifest.json`)
- [x] Contamination filter: removed 38 tables matching 1,141 benchmark header sets

## Phase 3: Baseline

- [x] Connect the repo's prompts and evaluators to vLLM
- [x] Write the corrected comparator; report both scores
- [x] Base baselines on all four final test sets: WikiSQL 0.326, Spider 0.725, WikiTQ 0.219, BIRD 0.364
- [x] Test the repo's few-shot prompt on base: WikiSQL 0.685, Spider 0.769, WikiTQ 0.344, BIRD 0.399
- [x] Test original-case prompt tables on WikiSQL: 0.142 and 0.287, so this does not explain the gap
- [x] Decide the evaluation prompt: the repo's zero-shot default, with few-shot reported beside it
- [x] Record the base gaps as unexplained (`docs/DECISIONS.md`)
- [x] Update the author email (`docs/author_email_draft.md`)
- [x] Author email cancelled: contacting the authors is not possible (decided by you, 2026-11-04). Unknowns stay recorded as unresolved in `docs/DECISIONS.md`
- [x] Record why the gate is not passed, and proceed

## Phase 4: Generator-Validator loop

- [x] Question prompt, requirement sampler and response parsing (`ts/gen/questions.py`), unit-tested; not yet run on the model
- [x] SQL and Spark-Scala prompts and code extraction (`ts/gen/prompts.py`, `ts/gen/code.py`), unit-tested; not yet run on the model
- [x] SQL-vs-Scala agreement check, stopping at the first disagreement (`ts/validate/exec_agree.py`); tested on real Spark
- [x] Run validation in parallel across pairs, and measure real programs' speed: 631 Scala runs in 1,002 s and 516 in 887 s at 36 workers
- [x] LoRA trainer with completion-only loss (`ts/train/sft.py`); 5 test steps ran on base at 18.7 GB
- [x] Rounds, arms, checkpoint and resume (`ts/loop.py`), unit-tested with fakes
- [x] Add per-round adapter evaluation to `ts/loop.py`, unit-tested with fakes; first real use is the running test
- [x] First real run on 300 tables per round, 2 rounds, finished: 42 and 31 pairs kept. Inspecting them found validator bugs, so it is superseded (`runs/old_test300_before_comparator_fix`)
- [x] Fix what that inspection found: SQL column types, null-only rows, booleans, Spark quote escaping, adapter names in the response cache
- [x] Classify all 160 `scala_failed` pairs from the first run by their Spark errors: all exit code 2; none needed a validator change
- [x] Rerun 300 tables per round, 2 rounds, on the corrected code: 41 and 43 pairs kept, first adapter scores recorded
- [x] Make GPU and port selectable in `ts/loop.py`, so that each GPU can run its own arm
- [x] Smoke test the no-validation arm alone on GPU 1: 20 tables, 2 rounds, 10 test cases per set; exit 0 in 17.7 minutes, 18.9 GB peak (log `logs/smoke_lane_gpu1.log`)
- [x] Limit power-cut losses in code: validation saved after each pair, training checkpoints about 12 times per adapter, files forced to disk
- [x] Test recovery with hard kills: training resumed from checkpoint-55 and finished 129 of 129 steps; validation kept 45 of 58 saved pairs and finished the rest (logs `logs/killtest_*.log`)
- [x] Inspect 50 kept and 50 rejected pairs by hand, on the corrected run: 39 kept pairs fit their question, 6 do not; 33 rejections are right, and 17 may have held usable SQL (`docs/RESULTS.md`)

## Phase 5: Core experiments

- [x] Vanilla arm: the baselines in `docs/RESULTS.md`; deterministic at temperature 0, so no further seeds run
- [x] Table-Specialist arm: 3 seeds, 3 rounds, 3,000 tables per round, restarting from vanilla each round (finished 2026-11-07, 03:12 IST)
- [x] No-validation arm: the same settings (finished 2026-11-07, 19:00 IST)

## Phase 6: Further experiments

- [ ] Ablations
- [ ] Error Detection on AutoTest
- [ ] Web-table corpus comparison (the repo's filenames suggest the authors used web tables)
- [ ] Instruct as a separate experiment, with its own fine-tuning runs
- [ ] Qwen arm and deck extensions. The three extensions from the mid-sem deck (v7, 12 Oct 2026), as listed in the Blueprint; which to run first is waiting for the user's decision (2026-11-07):
  - [ ] Extension 1: validator precision and recall against known-correct SQL
  - [ ] Extension 2: learn from rejected pairs with DPO. Chosen by the user on 2026-11-07 as the first novelty; design in `docs/DECISIONS.md`
    - [x] Code and unit tests: `ts/ext2.py`, `ts/train/dpo.py`, `tests/test_ext2.py` (8 tests; full suite 145 passed, 1 skipped)
    - [x] Smoke test: 39 questions, short training, 10 test cases per set; exit 0 at 20:20 IST on 2026-11-07 (log `logs/smoke_ext2.log`). Sampled SQL: 37-38 of 39 have code, 16-17 of 39 match the greedy program. 9 questions got a chosen program (4 greedy, 5 sampled), 5 DPO pairs, 106 Scala runs, DPO peak 20.8 GB
    - [x] DPO memory test at the 8 longest prompts (2,357 tokens): 21.4 GB peak; over 6 steps the reward margin rose from 0 to 2.10 and reward accuracy reached 1.0 (log `logs/smoke_dpo_memory_6steps.log`). First try failed: transformers 5 has no `warmup_ratio`, fixed by passing the ratio as `warmup_steps`
    - [x] Round 1 pilot, seeds 1-3 (GPU 0: seeds 1 and 3; GPU 1: seed 2). Seed 2 started 20:11 IST on 2026-11-07 in tmux `ext2_gpu1` (log `logs/ext2_seed2.log`). `ext2_gpu0` started 20:21 IST: seed 3 generation only (`--generate-only`, so GPU 0 works while seeds wait for the validation slot), then all of seed 1 (log `logs/ext2_gpu0_driver.log`). The rest of seed 3 goes on whichever GPU frees first. Seed 3 generation finished 22:17 IST (2,893 questions × 3 samples × 2 languages in about 2 h). Seed 2 crashed at 00:28 IST on 2026-11-08 with 2,476 of 2,893 questions labelled: a table SQLite cannot hold (`OverflowError`, the same 2 tables TS recorded as errors) was not caught in labelling. Seed 1 would hit 4 such tables, so it was stopped at 21 labelled questions. Fixed in `ts/ext2.py` (`_label_one` records the error and goes on, as `ts/loop.py` does), with 2 tests. Restarted 00:30 IST from the journals: `ext2_gpu1` runs seeds 2 then 3, `ext2_gpu0` runs seed 1. Seed 2 finished 04:28 IST, seed 1 07:23 IST (both exit 0; results in `docs/RESULTS.md`). Seed 3 labelled; training on GPU 1. GPU 0 idle from 07:23 IST **Finished 09:59 IST on 2026-11-08, all 3 seeds.** 3-seed mean: bestof4 7/1 columns above TS round 1 (+0.0064), dpo 0/8 (−0.0209); details in `docs/RESULTS.md`
    - [x] Dev sets for tuning (`ts/bench/dev.py`): WikiSQL 300, Spider 575, WikiTQ 300; no gold mismatches; test sets untouched
    - [x] DPO tuning on dev (`ts/ext2_tune.py`, 5 variants in `configs/base.yaml`). `tune_gpu0` started 08:01 IST on 2026-11-08: references TS, bestof4, first DPO, then v1-v3 (log `logs/ext2_tune_gpu0.log`). Finished 09:31 IST, exit 0: no variant above bestof4's dev mean (best v2 0.5600 vs 0.5618; results in `docs/RESULTS.md`). v4-v5 moved to GPU 0 as `tune_gpu0b`, started 09:31 IST (log `logs/ext2_tune_gpu0b.log`), finished 11:04 IST, exit 0: v4 0.5374, v5 0.5557. **Outcome: no variant above bestof4 (0.5618); DPO is not taken to the test sets.** Next step waits for the user: bestof4 rounds 2-3, or Extension 1
    - [ ] **On hold (user, 2026-11-08, 11:30 IST):** Extension 2 (bestof4 rounds 2-3). New extension ideas proposed to the user: adapter scaling and merging, test-time cross-language voting, repair of rejected pairs, 2-of-3 language voting, mixed-prompt training. Waiting for the user's choice
- Extension 3: repair rejected pairs (user's choice, 2026-11-08). Design in `docs/DECISIONS.md`
    - [x] `ts/repair.py` and `tests/test_repair.py` (8 tests; suite 158 passed, 1 skipped)
    - [x] Smoke test 1, 12 rejected pairs (`runs/repair_smoke_seed1`): pipeline works; 0 of 10 recovered, 6 of 10 greedy repairs copied the failed code. Added up to 3 retries at temperature 0.7
    - [x] Smoke test 2 with retries (`runs/repair_smoke2_seed1`): 1 of 10 recovered (disagree, Scala repaired), 9 of 10 produced new code, 19 Scala runs
    - [x] Pilot on seed 1's 300-table set: finished 12:00 IST, exit 0. 7 of 237 recovered, kept 41 → 48 (+17.1%); hand check 7 of 7 correct. **No-go** (needed +30%); details in `docs/RESULTS.md`
    - [ ] ~~If go: repair-TS round 1, 3 seeds~~ not started: pilot was no-go
    - [x] Instruct repairer option (`--repairer instruct`, `--plans-from`); smoke test on 8 pairs (`runs/repair_smoke_instruct_seed1`): 0 of 8 recovered, new code on the first attempt in 11 of 12 repairs; suite 159 passed, 1 skipped
    - [x] Instruct-repairer pilot, same 300 tables: finished 12:30 IST, exit 0. 23 of 237 recovered, kept 41 → 64 (+56.1%); hand check 21 correct, 2 doubtful, 0 wrong (91.3%). **Go**
    - [ ] Full run `ts/repair_ts.py`, 3 seeds (GPU 0: seeds 1, 3; GPU 1: seed 2). End-to-end smoke test (`runs/train_repairtssmoke_base_seed2`, 6 pairs, 2 steps, 5 cases per test set) passed, exit 0, 12:39 IST. **Full runs started 12:40 IST on 2026-11-08**: tmux `repairts_gpu0` (seeds 1 then 3, logs `logs/repairts_seed1.log`, `logs/repairts_seed3.log`) and `repairts_gpu1` (seed 2, `logs/repairts_seed2.log`). The GPU 0 session's final `exit=` line reports tee's status, so its seeds are checked by their `REPAIRTS` lines
        - Status at 17:43 IST: seed 1 repair finished (224 of 2,369 recovered, kept 426 → 650, +52.6%), trained, and 7 of 8 test cells scored; BIRD few-shot is scoring. Seed 2 repair validation at 2,130 of 2,398. Seed 3 not started (starts on GPU 0 after seed 1)
        - Seed 1 finished about 17:45 IST: mean of 8 0.5388 (TS round 1 0.5106). Seed 2 finished 19:44 IST, exit 0: 217 of 2,398 recovered, kept 388 → 605 (+55.9%), mean of 8 0.5584 (TS round 1 0.5159). Seed 3 started on GPU 0 about 17:45 IST, repair validation at 815 of 2,370 at 19:44 IST. GPU 1 idle: no job is queued for it until round 1 is decided
        - Seed 3 finished 22:16 IST, exit 0: 204 of 2,370 recovered (+48.2%), mean of 8 0.5555. **3-seed mean 0.5509, highest of all arms (TS round 1 0.5266, TS round 3 0.5435, NV round 1 0.5401)**; above TS round 1 on all 3 seeds
    - [x] Repair-TS round 1, 3 seeds: done 22:16 IST
    - [x] Rounds 2-3 (user approved 23:50 IST, 2026-11-08). Design and scheduling in `docs/DECISIONS.md`
        - [x] `ts/repair_loop.py` (GPU pool, parallel SQL/Scala training), `repair.run(gpu_slot=...)`, `tests/test_repair_loop.py` (3 tests; suite 163 passed, 1 skipped)
        - [x] Smoke test (`runs/train_repairtssmoke_base_seed2`, rounds 2-3, 12 tables, 6 repairs, 2 steps, 5 cases per test set): started 23:55 IST in tmux `rloop_smoke`, finished 00:30 IST on 2026-11-09, exit 0. Both rounds completed; SQL and Scala adapters trained on GPUs 0 and 1 at once; round 3 generated with round 2's adapters; round 3 recovered 1 of 6 repaired pairs
        - [x] Full rounds 2-3, seeds 1-3 at once (all finished, exit 0; seed 3 last at 14:39 IST, 2026-11-10; repair-TS round 3 3-seed mean 0.5856 vs TS round 3 0.5435): **started 00:30 IST on 2026-11-09** (tmux `rloop_seed{1,2,3}`, logs `logs/repair_loop_seed{s}.log`). Each seed prints a `REPAIRTS_ROUND` line per round and an `exit=` line at the end
            - Status at 07:49 IST, 2026-11-09: no errors. Round 1 Scala adapters done for all 3 seeds (about 40 min each). Round 2 generation done for all 3 (about 31 min each). Round 2 validation: seed 2 done (3 h 12 min, 2,916 pairs), seed 1 at 2,711 of 2,918, seed 3 waiting for the validation slot; seed 2 waiting for the slot to plan its repairs. GPUs idle: Spark validation (one seed at a time) is the bottleneck. Projected finish of all rounds about 12:00 IST on 2026-11-10
            - Status at 16:58 IST, 2026-11-09: seed 2 round 2 finished 16:40 IST (mean of 8 0.5543; TS round 2 0.5109); seed 2 round 3 generating on GPU 0. Seed 1 round 2: repair done (140 of 2,134 recovered, +20.3%), Scala adapter done, SQL adapter training on GPU 1. Seed 3 round 2: repair validation at 558 of 2,106. No errors. Projected finish of all rounds about 13:00 IST on 2026-11-10
            - Status at 07:59 IST, 2026-11-10: round 2 done for all 3 seeds (**3-seed mean 0.5713**; TS round 2 0.5286, TS round 3 0.5435). Round 3: all 3 seeds generated and validated; seed 2 repair validation at 827 of 2,010; seeds 1 and 3 waiting for the validation slot. No errors. Projected: seed 2 round 3 about 11:00 IST, last seed about 14:30 IST
            - Seed 2 finished all rounds 11:18 IST, 2026-11-10, exit 0 (round 3 mean of 8 0.5801; TS round 3 0.5244). Seed 1 round 3: repair done, both adapters training. Seed 3 round 3: repair validation at 345 of 1,905. Projected: seed 1 about 13:30 IST, seed 3 about 15:00 IST
            - Seed 1 finished all rounds 13:09 IST, exit 0 (round 3 mean 0.5864; TS round 3 0.5660). Seed 3 round 3: repair done, adapters training on both GPUs; projected about 15:00 IST
  - [ ] Extension 3: 0.5-3B specialists trained on the 8B-validated data

## Phase 6b: Match the paper, then improve (proposed 2026-11-10)

- [x] Analysis of the paper (HTML v2), the repo and our results; plan in `docs/PLAN_NEXT.md`
- [x] User approved the plan, 2026-11-10 ("Go ahead"); my recommended answers taken (Instruct for matching, established methods only, A3 runs)
- [x] A1 Instruct vanilla gate (`ts/a1_instruct_gate.py`): started 21:39 IST in tmux `a1_gate`, GPU 0, log `logs/a1_instruct_gate.log` (first try failed at import, run as a script; rerun as a module). At 21:46 IST: WikiSQL z 0.518 (paper 0.525), WikiSQL f 0.798, Spider z 0.7913 (paper 0.670), Spider f 0.7963, WikiTQ z 0.289 (paper 0.300). Finished 22:42 IST, exit 0: WikiTQ f 0.423, BIRD z 0.4022 (paper 0.388), BIRD f 0.4187. **Gate passed on WikiSQL, WikiTQ, BIRD; Spider +0.121. Go to A2**
- [x] A2 code: `TS_PROFILE=instruct` in `ts/loop.py`; suite 166 passed, 1 skipped
- [x] A2 smoke test 1 (`train_tablespecialist_instruct_seed1_smoke`, 20 tables): exit 1 at 22:49 IST. Generation and validation worked (20 of 20 SQL and Scala extracted), but 0 of 20 pairs were kept (always_empty 11, disagree 4, scala_failed 4, sql_failed 1), so training had no examples. Base kept 14.7% in round 1, so 0 of 20 can be chance; always_empty at 55% (base 21%) is watched in the full run
- [x] A2 smoke test 2 (`..._smoke2`, 60 tables, 2 rounds): started 22:49 IST, finished 23:14 IST, exit 0. Kept 4 of 60 (round 1) and 3 of 60 (round 2); both rounds trained and scored in chat form. Instruct keeps far fewer pairs than base (5-7% against 14.7%), mostly always_empty
- [x] GPU pool for the core loop (`ts.loop --gpu-pool`: each GPU step takes a free GPU, validation takes the shared validation lock) and for `ts.sc_run`; defaults unchanged; suite 166 passed, 1 skipped
- [x] B1 test restarted under the pool at 23:16 IST (tmux `sc_test`); finished cells are skipped, the cell in progress is redone from the response cache
- [x] A2 smoke test 3, pool path (`..._smoke3`, seed 2, 40 tables, 1 round): finished 23:27 IST, exit 0; the two jobs held different GPUs; validation took the shared lock; kept 6 of 40
- [x] **A2 full, Instruct TS seeds 1-3 × 3 rounds, and A3, Instruct NV seed 1 × 3 rounds: started 23:28 IST, 2026-11-10** (tmux `its_seed{1,2,3}`, `inv_seed1`; logs `logs/instruct_ts_seed{s}.log`, `logs/instruct_nv_seed1.log`; run IDs `train_tablespecialist_instruct_seed{s}`, `train_novalidation_instruct_seed1`). All share the GPU pool with B1. Projected finish: Thu 12 Nov, about 10:00-14:00 IST. Round 1 done (2026-11-11): TS seed 1 zero-shot mean of 4 0.6280, TS seed 2 0.6096, NV seed 1 0.6162 (paper TS 0.5983); TS seed 3 in round 1 training at 15:35 IST. **TS seed 1 finished 2026-11-12, 05:30 IST, exit 0** (zero-shot mean of 4: r1 0.6280, r2 0.6346, r3 0.6156). **TS seed 2 finished 2026-11-12, 11:00 IST, exit 0** (r1 0.6096, r2 0.6168, r3 0.5961). **NV seed 1 finished 11:02 IST, exit 0** (r1 0.6162, r2 0.6094, r3 0.5852; TS beats NV at round 3 on both finished seeds). B1 restarted at 11:03 IST in tmux `sc_after`: Instruct vanilla and TS seeds 1-2 first (log `logs/sc_instruct.log`), then the BIRD re-scoring (log `logs/sc_birdseq.log`); one B1 job at a time, so that TS seed 3 always has a GPU. TS seed 3 B1 runs after its round 3. **TS seed 3 finished 12:54 IST, exit 0; A2 and A3 complete.** 3-seed round 3 zero-shot mean of 4: 0.6113 ± 0.0135 (paper 0.5983); WikiSQL, WikiTQ and BIRD within 0.032 of the paper. `sc_after` will include seed 3 (each round is checked when its turn comes). BIRD re-scoring moved to its own tmux `sc_birdseq`, started 12:55 IST on the free GPU
- [x] B1 code: `ts/eval/self_consistency.py`, `ts/sc_run.py`, tests
- [x] B1 smoke test (20 dev cases): started 21:42 IST in tmux `sc_smoke`, finished 21:45 IST, exit 0; the vote changed 1 of 60 answers (Spider, wrong to right)
- [x] B1 dev tuning: started 21:46 IST, finished 22:31 IST, exit 0. T 1.0 wins (dev mean 0.6520 vs greedy 0.6225)
- [x] B1 test sets at T 1.0: started 22:32 IST in tmux `sc_test` (restarted under the pool 23:16 IST), finished 2026-11-11, 15:35 IST, exit 0, 56 of 56 cells. Mean of 8, greedy → vote: vanilla 0.4767 → 0.5015, TS r3 0.5414 → 0.5580, repair-TS r3 0.5833 → 0.6000. Repair-TS + vote, zero-shot mean of 4: 0.6051 (paper TS 0.5983). See RESULTS
- [x] Check why BIRD greedy in `self_consistency.py` is 3 cases below `run_baseline` (found and fixed 2026-11-12, see below) (likely timeouts under 16 parallel scorers)
- [x] BIRD cause found (2026-11-12): the 3 cases return 20,198-49,729 rows and hit the 5 s SQL timeout under 16 scoring threads; `run_baseline` scores on one thread. Fix: `--score-workers` in `self_consistency.py`, 1 for BIRD in `sc_run.py`; suite 166 passed, 1 skipped
- [x] Re-score the 14 base BIRD cells on one thread under new run IDs `final_sct10seq_*` (old files kept); **finished 17:57 IST, exit 0; greedy now matches `run_baseline` exactly. Corrected: repair-TS + vote, zero-shot mean of 4, 0.6067; mean of 8, 0.6015.** Earlier notes:: started 05:39 IST, 2026-11-12, tmux `sc_birdseq`, log `logs/sc_birdseq.log`, GPU pool. It never got a GPU; stopped 08:26 IST (Ctrl-C) so that A2 goes first. Restart after A2/A3
- [ ] B1 on Instruct (plan step 4): `sc_run` is profile-aware (Instruct vanilla, Instruct TS r3, chat, repo system message, 256 tokens; skips rounds not yet finished). Smoke test (10 cases per cell, Instruct TS seed 1): started 05:32 IST, finished 05:39 IST, exit 0, 8 of 8 cells, chat, 256 tokens, BIRD on one thread; the vote changed 6 answers from wrong to right and none from right to wrong
- [x] Repair-TS on Instruct, code (plan step 4; A2 matched the paper): `ts/repair_ts.py` and `ts/repair_loop.py` follow `TS_PROFILE` (source `train_tablespecialist_{profile}_seed{s}/round1`, run ID `train_repairts_{profile}_seed{s}`) and the GPU pool; `repair_loop` runs round 1 itself when it is missing. Base runs unchanged (their round 1 exists). Suite 166 passed, 1 skipped
- [x] Repair-TS on Instruct, smoke test (`train_repairtssmoke_instruct_seed2`, round 1 from the 40-table TS smoke run `_smoke3`, round 2 on 60 tables, 40 repairs, 5 SFT steps, 20 test cases): started 17:59 IST, 2026-11-12, tmux `irts_smoke`, log `logs/instruct_repairts_smoke.log`. Finished 18:36 IST, exit 0, no Tracebacks: round 1 kept 6 of 40 and recovered 2 of 34 rejected pairs (+33%); round 2 kept 5 of 60 and recovered 1 of 39; both rounds trained and scored in chat form
- [x] **Repair-TS on Instruct, full, seeds 1-3 × 3 rounds: started 18:36 IST, 2026-11-12** (tmux `irts_seed{1,2,3}`, logs `logs/instruct_repairts_seed{s}.log`, run IDs `train_repairts_instruct_seed{s}`, GPU pool, shared with `sc_after`). About 30-40 h, so projected for Sat 14 Nov, between about 01:00 and 11:00 IST. **2026-11-13, 09:10 IST:** round 1 done for all seeds (zero-shot mean of 4: 0.6275, 0.6252, 0.5941). Round 2: seed 3 is validating; seeds 1-2 wait for the validation lock (repair checks). Both GPUs are idle: CPU validation is the limit. **Revised projection: Sat 14 Nov, about 09:00-14:00 IST** **2026-11-14, 05:53 IST: seed 1 finished all 3 rounds (exit 0)**; round 3 mean of 8 0.6073, zero-shot mean of 4 0.6135 (TS seed 1 r3: 0.6118 / 0.6156). Seed 2 is validating round 3; seed 3 waits for the validation lock. **09:25 IST:** seed 3 is checking its round-3 repairs (validation lock, 993 of 2,346, about 1.4 h left), then trains and scores (about 2.4 h): projected about 13:30 IST. Seed 2 has generated its repairs and waits for the lock (about 2.5 h of checks after seed 3, then 2.4 h): **projected about 18:00 IST.** **11:43 IST: seed 3 finished all 3 rounds (exit 0)**; round 3 mean of 8 0.6118 (TS seed 3 r3: 0.6169). Seed 2's repair checks finished early; it is training round 3 on GPU 1 (from 11:32 IST): projected about 14:00 IST. **13:00 IST: seed 2 finished all 3 rounds (exit 0); all seeds done.** 3-seed round 3: zero-shot mean of 4 0.6122, vs TS r3 mean of 8 −0.0014 (p 0.55): no difference.
- [x] **B1 on Instruct repair-TS r3, seed 1** (finished 09:15 IST, exit 0; smoke OK; greedy matches all 8 training-run cells; vote mean of 8 0.6214 against TS r3 + vote seed 1 0.6282) (`ts/sc_run.py` now lists `instruct_repairts_seed{s}`): smoke test (5 cases) then the full run, chained, tmux `sc_irts1`, log `logs/sc_instruct_repairts_seed1.log`, run IDs `final_sct10_instruct_repairts_seed1_*`, started 05:53 IST, 2026-11-14. Seeds 2-3 follow as each finishes.
- [x] **B1 on Instruct repair-TS r3, seed 3** (finished 14:44 IST, exit 0; greedy matches all 8 cells; vote zero-shot mean of 4 0.6377, mean of 8 0.6292): tmux `sc_irts3`, log `logs/sc_instruct_repairts_seed3.log`, run IDs `final_sct10_instruct_repairts_seed3_*`, started 11:44 IST, 2026-11-14 (no smoke: same code path as seed 1's passing smoke and full run). About 3.5 h.
- [x] **B1 on Instruct repair-TS r3, seed 2** (finished 15:20 IST, exit 0; greedy matches all 8 cells; vote zero-shot mean of 4 0.6288, mean of 8 0.6219). **All planned GPU runs are finished; no tmux sessions remain.**: tmux `sc_irts2`, log `logs/sc_instruct_repairts_seed2.log`, run IDs `final_sct10_instruct_repairts_seed2_*`, started 13:00 IST, 2026-11-14 (GPU freed by seed 2). About 3.5 h; with seed 3 running in parallel, both done about 16:30 IST.
- [x] **Final analysis** (15:25 IST): `ts.analysis.significance` (23 comparisons, `results/analysis/significance.{json,md}`) and `ts.analysis.results_table` (`results/analysis/results_table.md`) rerun on the complete data. Paper TS mean now prints 0.5983 (exact decimal mean).
- [ ] Write-up: discrepancy log (each difference from the paper and its cause), compute-cost table (GPU hours; the vote's ×9 generation cost), one final presentation table, release scripts (Phase 7).
- [x] B1 on Instruct, full (**finished 19:58 IST, 2026-11-12, exit 0, 32 cells; Instruct TS r3 + vote, zero-shot mean of 4: 0.6277 ± 0.0097, paper 0.5983**): started 05:39 IST, 2026-11-12, tmux `sc_instruct`, log `logs/sc_instruct.log`: Instruct vanilla and TS seed 1. Seeds 2-3 are skipped until their round 3 finishes, then rerun with the same command (finished cells are skipped). Stopped 08:26 IST (Ctrl-C), with 7 of 8 Instruct vanilla cells done. It had held GPU 0 while Instruct TS seed 2 waited about 2.5 h to train round 3. Seed 2 took GPU 0 at 08:26 IST. Restart after A2/A3, for all Instruct models at once

- [x] **Significance tests** (`ts/analysis/significance.py`, 2026-11-13, 09:43 IST): paired bootstrap over test cases plus per-seed differences, saved predictions only. All main base and Instruct claims have p < 0.0001; Instruct repair-TS r1 vs TS r1 is not significant (p 0.13). Pending comparisons fill in on rerun
- [x] **Same-data control** (finished 2026-11-13, 22:00 IST, exit 0. Base: control 0.5385 mean of 8, between TS r1 0.5266 and repair-TS r1 0.5509, both gaps p < 0.0001. Instruct: control 0.6231, above TS r1 0.6152 (p 0.0006) and above repair-TS r1 0.6117 (p < 0.0001); see RESULTS.md) (`ts/data_control.py`): TS round 1 kept pairs plus N extra kept pairs from another seed's TS round 1 (tables the seed never used), N = repair-TS round 1's recovered count (base: 224, 217, 204). Training and scoring only; no generation or validation, so it does not compete with repair-TS for the CPUs. One seed at a time, so one GPU stays free. Smoke test (`train_tsdatasmoke_base_seed1`, 5 steps, 20 cases) started 09:40 IST in tmux `tsdata_smoke`, log `logs/tsdata_smoke.log`; finished 09:45 IST, exit 0 (650 examples, as repair-TS seed 1). **Full run started 09:46 IST, tmux `tsdata`, log `logs/tsdata.log`: base seeds 1-3, then Instruct seeds 1-3 (Instruct N = 74, 75, 65), run IDs `train_tsdata_{base,instruct}_seed{s}`.** About 1-1.5 h per seed, so about 6-9 h
- [x] **Complete per-dataset results table** (2026-11-13): `ts/analysis/results_table.py` → `results/analysis/results_table.md` (every approach, both models, both prompts, mean ± SD over seeds). Rerun after Instruct repair-TS round 3 finishes, together with `ts.analysis.significance`.

## Phase 7: Analysis and report

- [x] Results table with mean ± std over seeds, beside the paper's numbers (`docs/RESULTS.md`, "Final results table, all approaches", 2026-11-10)
- [ ] Discrepancy log tying each difference to a known cause
- [ ] Release configs, scripts, ID lists and `make reproduce` for one seed

## Documentation

- [x] `docs/TASKS.md`, `docs/DECISIONS.md`, `docs/RESULTS.md`, updated at every step
- [ ] Update the README scope to match `configs/base.yaml`
- [ ] Add the reduced-scope reversal to the Blueprint, if wanted


## Phase 8 (approved 2026-11-14, 17:15 IST): the paper's other tasks

**WSL crash, 2026-11-15 about 12:45 IST** (Wsl/Service/E_UNEXPECTED; the user ran `wsl --shutdown`). All jobs were killed; the last log lines are from 12:41 IST. **Resumed 13:01 IST with the user's approval.** First, the interrupted round folders, the partial `preds.jsonl`, the response cache and the five logs were copied to `backups/crash_20261115_1245/` (1.4 GB, verified with diff/cmp), and the 36 stopped sandbox containers were removed. Then each job got its own tmux session, appending to its existing log after a `RESUME` line:
- `edts_seed1` (GPU 0): training resumed from checkpoint-210, at step 212 of 306 at 13:04 IST
- `edts_seed2` (GPU 1)
- `nlscala_score2`: 11 cells, 241 programs to run
- `nlscala_score3`: 28 cells, 8,395 programs to run; waits for the Spark slot

Seed 3 starts when seed 1 or seed 2 finishes (one job per GPU). **`nlscala_score2` finished 14:08 IST, exit 0: all 28 final-round zero-shot cells are scored** (table in RESULTS.md). Seed 1 round 2 finished 14:05 IST (F1 ST 0.1954, RT 0.2742; 3,537 examples); round 3 is running. **Seed 1 finished 17:26 IST, exit 0** (round 3 F1 ST 0.2655, RT 0.3191). Seed 3 started 17:26 IST on GPU 0 (tmux `edts_seed3`, log appended). Seed 2 round 2 done (F1 ST 0.2609, RT 0.3009); round 3 running. Few-shot Scala: all 8,395 programs run; cells being written. **BIRD write slowness fixed 18:05 IST** (quadratic comparison in `ts/eval/compare.py`; identical verdicts on all 45,930 scored cases; see DECISIONS.md). The running `nlscala_score3` keeps the old code; the rounds 1-2 scoring will use the fix and is now projected to finish Mon about 21:00 to Tue about 01:00 IST State checked at 12:55-13:10 IST. All checked files parse, and the response cache passes SQLite's quick_check. Details are in RESULTS.md, "WSL crash, 2026-11-15":
- **Error detection.** Seed 1: round 1 done; round 2 training stopped at step 210 of 306, with an intact checkpoint. Seed 2: round 1 done; round 2 errors generated, validation not started. Seed 3: round 1 errors generated; validation at 3,143 of 32,820 requests, which are cached
- **NL-to-Scala zero-shot (`nlscala_score2`).** All 15,420 programs ran. 17 of 27 cells were written. One cell has a partial `preds.jsonl` (959 of 1,000 lines) and no result file; 9 cells were not written
- **NL-to-Scala few-shot (`nlscala_score3`).** About 2,600 of 12,087 programs are cached; no cell was written

Analysis of which other task to reproduce next and apply repair-TS to (full reasoning in RESULTS.md, "Other tasks in the paper"). Ranked easiest first:

1. **NL-to-Scala** (recommended): no new data generation or training. Every TS / NV / repair-TS round already trained `adapter_scala` (61 adapters). Needs a test-time Scala prompt (repo `nss` generator), a scorer (our Spark sandbox + the repo's `nss_evaluator` comparison against gold SQL results; our single-table test cases already have `data.csv`) and CPU time (about 1.5 h per model per prompt).
2. NL-to-R: same loop with SQL vs R validation; new R generation and R sandbox; full 3-round runs (about 30-40 h compute per profile).
3. Schema matching: no code execution, but the test sets (DeepM 42, WikiData 24, HXD 468) are not in the repo or linked; repair-TS has no execution feedback to use.
4. Error detection: AutoTest is downloaded but its counts do not match the paper's; F1 about 0.1; repair-TS does not apply.
5. Table-QA: validation method not described; FinQA and TableBench not local.
6. Data transformation: the paper's "hard" task; TDE and Transform-Text not local; scores 0.05-0.14.

**Approved plan (Instruct only):**
- [x] NL-to-Scala scorer `ts/eval/scala_eval.py` (11 tests pass; gold cross-checked against the SQL runs: identical, except one BIRD case whose gold uses JULIANDAY('now')). First smoke test void (sandbox settings not sourced; see DECISIONS.md); rerun `smoke_nlscalar2_*` finished 17:55 IST, exit 0: harness works, 0.579 programs/s (repo `nss_evaluator` rules; dedup cache; generation and Spark scoring as separate phases) + unit tests + smoke test on a few cases per test set, with a Spark throughput measurement
- [x] **Running:** generation for the 7 final-round models started 17:56 IST, tmux `nlscala_gen_a` (vanilla, TS r3 seeds 1-3; log `logs/nlscala_gen_final_a.log`) and `nlscala_gen_b` (repair-TS r3 seeds 1-3; log `logs/nlscala_gen_final_b.log`), one GPU each; `nlscala_gen_b` finished 19:25 IST, exit 0. `nlscala_gen_a` froze in the machine stall (19:37 to 03:03 IST, see RESULTS.md) and resumed by itself; it finished 03:14 IST, exit 0 (all 7 final-round models, 56 cells). Generation for the other 12 models (TS and repair-TS rounds 1-2) started 03:15 IST, tmux `nlscala_gen_c`, log `logs/nlscala_gen_rest.log`, one GPU (the other is for error detection). Cells finished before the stall are valid: the client raises on a failed request and never stores an empty answer. Then the other 12 models
- [x] Generate Scala code for 19 Instruct models (all 152 cells; `nlscala_gen_c` finished 09:22 IST, exit 0) (vanilla; TS and repair-TS, seeds 1-3, rounds 1-3) × 4 test sets × 2 prompts, both GPUs (about 6 h)
- [x] ~~Spark scoring `nlscala_score1`~~ (started 18:50 IST; 1,686 programs done before the stall): stopped by me at 03:07 IST, before writing any cell, because it held the 36 stall timeouts in memory. No cell files were written; all finished programs stay cached
- [x] Spark scoring, zero-shot (`nlscala_score2`, finished 2026-11-15 14:08 IST, exit 0, all 28 cells), of the 27 final-round cells generated by 03:07 IST (all except TS r3 seed 3 BIRD): started 03:07 IST, tmux `nlscala_score2`, log `logs/nlscala_score_final_zs2.log`; 17,164 unique programs, 15,420 to run (these include the 36 stall re-runs), about 7.5 h
- [x] Spark scoring, few-shot, final rounds: `nlscala_score3` finished 18:16 IST, exit 0 (28 cells)
- [x] **Finished 2026-11-16 06:04 IST, exit 0 (all 152 cells; tables and paired bootstrap in RESULTS.md: repair-TS ahead of TS at every round, p < 0.0001, all seeds positive).** Spark scoring of all remaining cells (rounds 1-2 of TS and repair-TS, seeds 1-3, both prompts: 96 cells, 38,909 unique programs, 29,586 to run, about 11 h at 0.73/s): started 18:17 IST, tmux `nlscala_score4`, log `logs/nlscala_score_rounds12.log`; uses the compare.py speed fix. Projected Mon about 05:30 IST
- [x] Score zero-shot for the final rounds first, then few-shot, then the rest (order kept)
- [x] Error-detection scorer `ts/eval/ed_eval.py` (6 tests pass; AutoTest cases = the paper's 1,126 and 1,081 exactly); smoke test finished 17:46 IST, exit 0
- [x] **Error-detection vanilla gate** (finished 17:55 IST, exit 0): F1 0.0659 (ST) and 0.1111 (RT) against the paper's 0.071 and 0.108: passes
- [x] **Error-detection TS loop** `ts/ed_loop.py` (repo typo + semantic-error prompts, Algorithm 2 with 5 shuffles, clean columns, train from vanilla each round, validator = previous round's adapter; 6 tests pass, suite 189 passed): smoke test (seed 1, 2 rounds, 40 tables, 5 SFT steps, 20 test columns) started 18:00 IST, tmux `edts_smoke`, log `logs/edts_smoke.log`. Round 1 finished: 73 errors generated, 60 usable, 18 kept (4 typo, 5 semantic, 9 clean), adapter trained, test F1 0 on 20 columns (expected after 5 SFT steps). Round 2 failed at 19:55 IST, exit 1: its vLLM server did not start within 20 minutes because the machine stalled. Rerun as tag `smoke2` (new run directory; `train_edtssmoke_instruct_seed1` left as it is): started 03:07 IST, tmux `edts_smoke2`, log `logs/edts_smoke2.log`, GPU 1. **Smoke rerun finished 03:28 IST, exit 0**: both rounds ran end to end; round 2 used round 1's adapter as validator. Round 1 kept 4 typo + 5 semantic + 9 clean of 27 + 33 + 20; round 2 kept 0 + 0 + 20 of 24 + 34 + 20. The 5-step smoke adapter always answers NULL (it also predicts nothing on the 20 test columns), so its validator keeps only clean columns. Expected at this scale; in the full run, check after round 1 that the adapter still predicts errors, or later rounds would train on clean columns only
- [x] **Running: full error-detection TS, seeds 1 and 2 × 3 rounds** (3,000 tables per round, 1,000 negative tables): started 03:28 IST, tmux `edts_seed1`, `edts_seed2`, logs `logs/edts_seed{1,2}.log`, run IDs `train_edts_instruct_seed{1,2}`; share the GPU pool with `nlscala_gen_c`. Seed 3 started 09:23 IST on the GPU freed by `nlscala_gen_c` (tmux `edts_seed3`, log `logs/edts_seed3.log`). Status 08:30 IST: seed 1 round 1 done (F1 ST 0.1397, RT 0.1989; paper TS 0.136, 0.161), round 2 running; seed 2 round 1 in validation (19%). Spark scoring at 10,000 of 15,420; `nlscala_gen_c` at 80 of 96 cells
- [x] Error detection on the GPUs while Spark scores: first the vanilla gate on AutoTest (Instruct), then TS rounds, then repair-ED with best-of-N and same-data controls
- [x] **Error-detection TS seed 2 finished 20:39 IST, exit 0** (round 3 F1 ST 0.2909, RT 0.2857; 4,277 examples). GPU 1 free
- [x] **Repair-ED** `ts/repair_ed.py` (variants repair, resample = best-of-2 control, samedata = same-data control; design in DECISIONS.md; 5 tests in `tests/test_repair_ed.py`, suite 196 passed, 1 skipped): **smoke test started 20:45 IST**, tmux `edrepair_smoke`, log `logs/edrepair_smoke.log` (repair seed 1, 2 rounds on the 40-table `edtssmoke2` run, then resample 1 round, then samedata on the real TS round 1 data; 5 SFT steps, 20 test columns; run IDs `train_ed{repair,resample,data}smoke_instruct_seed1`). **Smoke test finished 21:15 IST, exit 0**: all three variants ran end to end (repair round 1: 51 rejected errors retried, 44 usable, 9 recovered; round 2 used its own round 1 adapter as validator; resample recovered 5; samedata 2,188 TS kept + 9 donor errors). One repair prompt and reply checked by hand. **Full repair-ED started 21:16 IST**: tmux `edrepair`, log `logs/edrepair.log`, seeds 1-3 x 3 rounds in one process (one GPU at a time), run IDs `train_edrepair_instruct_seed{1,2,3}`. About 11-12 h per seed on one GPU. Controls (resample and samedata, round 1, seeds 1-3) start on the GPU that error-detection TS seed 3 frees (about Mon 05:00 IST); samedata for a seed needs that seed's repair round 1. **Error-detection TS seed 3 finished 2026-11-16 04:32 IST, exit 0** (round 3 F1 ST 0.2791, RT 0.2889): all three TS seeds done. Repair-ED seed 1 round 1 done (506 recovered; F1 ST 0.1545, RT 0.2353); round 2 running. **Controls started 04:33 IST on GPU 0**: tmux `edcontrols`, log `logs/edcontrols.log`, resample seeds 1-3, then samedata seeds 1-2 (samedata seed 3 waits for repair seed 3 round 1, about Tue 03:00). **`edcontrols` finished 14:39 IST, exit 0**: resample seeds 1-3 and samedata seeds 1-2 done (samedata F1 ST/RT: seed 1 0.1561/0.2280, seed 2 0.2404/0.2258)

**Projected finish times (IST, as of 2026-11-15 03:15, after the 7.5 h machine stall; Spark 0.57 programs/s, reuse about 13%):**
| Item | Projected |
|---|---|
| Scala generation, 7 final-round models | Sun about 03:30 |
| Error-detection loop smoke test (rerun) | Sun about 04:00 |
| NL-to-Scala, final rounds, zero-shot (headline vs paper) | Sun about 11:00 (TS r3 seed 3 BIRD about 12:00) |
| NL-to-Scala, final rounds, few-shot | Sun about 23:00 |
| Scala generation, other 12 models | Sun about 04:00-12:00 (GPU, shared with error detection) |
| NL-to-Scala, all 19 models, both prompts | Tue about 12:00-16:00 |
| Error-detection TS, 3 seeds x 3 rounds | Mon morning to afternoon (depends on kept-column counts) |
| Repair-ED + best-of-N + same-data control | built and smoke-tested Sun 21:15; repair seeds 1-3 on one GPU: seed 1 Mon about 10:00, seed 2 Mon about 23:00, seed 3 Tue about 12:00 (earlier if the second GPU is used for it after Mon 05:00); controls Mon 05:00 to about Tue 03:00 on the other GPU (as of 2026-11-15 21:20 IST) |

**Status 2026-11-16 08:25 IST.** Error detection: TS done (3 seeds x 3 rounds); repair-ED seed 1 done, seed 2 running (projected Mon about 19:30 IST), seed 3 Tue about 07:00; best-of-2 control seed 1 done, seeds 2-3 Mon about 10:45 and 13:50; same-data control seeds 1-2 Mon about 16:20 and 18:50, seed 3 Tue about 02:30 (waits for repair-ED seed 3 round 1); ED significance tests and write-up Tue morning.
- [x] **NL-to-R (Phase 9; approved 2026-11-16 08:40 IST, "without hampering the current running experiments of error detection")**. Design in DECISIONS.md, "Phase 9".
  - [x] R sandbox image built (`sandbox/Dockerfile.r`: r-base 4.4.2 digest + Debian dplyr; R 4.6.1, dplyr 1.2.1), registered in `configs/sandbox.env`; one R run takes about 2.3 s (Spark: about 50 s)
  - [x] Code: `ts/validate/r_agree.py` (SQL vs R, Algorithm 3), `ts/r_loop.py` (TS and repair-TS rounds, run IDs `train_rts_instruct_seed<s>`, `train_rrepairts_instruct_seed<s>`), `ts/eval/r_eval.py` (repo `nr_evaluator` rules, run IDs `final_nlr_*`), R prompts in `ts/gen/prompts.py`, `ts/gen/code.py`, `ts/repo.py`. 8 tests in `tests/test_r.py`; suite 204 passed, 1 skipped
  - [x] CPU check of the scorer in the real sandbox: hand-written R for 3 WikiSQL test cases scored as expected (2 correct, 1 deliberately wrong)
  - [x] Smoke test (GPU 0 only): **finished 15:02 IST, exit 0** (all stages ran; TS round 1 kept 3 of 30, round 2 8 of 30; vanilla smoke accuracies 0.1-0.9 on 10 cases; R execution failures are genuine model errors, e.g. unquoted column names with spaces). It exposed a bug: repair feedback kept only the first line of rlang/dplyr errors (`Error in filter():`) and dropped the cause; fixed in `r_loop.r_messages` (backup `backups/r_loop.py.before_rmessages_20261116`), new test, suite 205 passed, 1 skipped. Only the repair arm uses it. Started 14:40 IST, tmux `nlr_smoke`, log `logs/nlr_smoke.log`, run IDs `train_{rts,rrepairts}smoke_instruct_seed1`, `smoke_nlr_*`. `edcontrols` freed GPU 0 at 14:39 IST, earlier than planned.: TS and repair-TS, seed 1, 2 rounds, 30 tables, 5 SFT steps; vanilla test generation and scoring on 10 cases per test set
  - [x] Full runs: **TS-R seeds 1-3 started 15:06 IST, 2026-11-16, GPU 0, tmux `nlr_ts`, log `logs/nlr_ts.log`**; Seed 1 round 1 done 16:22 IST (kept 321 of 2,978; 76 min), round 2 done 20:10 IST (kept 525). Round 2 lost about 1.8 h: during TS-R's CPU-only validation (17:02-17:14 IST) repair-ED's gpu_slot, which tries GPU 0 first, took GPU 0 and held it until 19:03 while GPU 1 sat idle. Fix: `TS_R_HOLD_GPU=1` keeps the run's GPU through validation (backup `backups/r_loop.py.before_holdgpu_20261116`); `nlr_ts` restarted with it at 20:12 IST during round 3 generation (cached responses, nothing lost). Also `TS_ED_GPUS` added to `ts/repair_ed.py` (backup `backups/repair_ed.py.before_edgpus_20261116`) so that samedata seed 3 runs with `TS_ED_GPUS=1`. Suite 205 passed, 1 skipped. vanilla test cells; TS-R seeds 1-3 on GPU 0 from Mon evening; repair-TS-R seeds 1-3 on GPU 1 after error detection leaves it (Tue morning); test generation and scoring of the 19 models
  - Scheduling: error-detection same-data control seed 3 needs repair-ED seed 3 round 1 (about Tue 00:00). With GPU 0 given to NL-to-R overnight, it runs on GPU 1 right after repair-ED seed 3 (about Tue 07:00 to 09:30) instead of about Tue 00:00 to 02:30 on GPU 0. No running error-detection job is affected. **The user approved this at 10:05 IST ("go ahead, NL-to-R can use GPU 0 overnight").**

**2026-11-16, 11:40 IST: LaTeX research report built** (`report/`, `report/main.pdf`, 121 pages). Every results table and plot is generated from `results/` and `runs/` by `report/tools/build_tables.py`; 24 of 24 headline values cross-check against this file's RESULTS counterpart (Appendix G). Compiled with `report/build.sh` in a pinned TeX Live 2025 Docker image (nothing installed on the host). Rerun `report/build.sh` after repair-ED, its controls and NL-to-R finish; the text describing running experiments is dated 2026-11-16 and needs a manual update then.

**2026-11-17, 01:17 IST:** repair-ED (tmux `edrepair`) finished, exit 0; round 3 F1 mean ST 0.2586, RT 0.3398 (TS 0.2785, 0.2979). 01:17 IST: tmux `edsd3` started on GPU 1: samedata seed 3 (`TS_ED_GPUS=1`, log `logs/edcontrols.log`), then repair-TS-R seeds 1-3 (`TS_R_GPUS=1 TS_R_HOLD_GPU=1`, log `logs/nlr_repairts.log`, run IDs `train_rrepairts_instruct_seed{1,2,3}`). Remaining for error detection after samedata: significance tests and write-up.

**2026-11-17, 02:48 IST:** same-data control seed 3 finished, exit 0 (F1 ST 0.1792, RT 0.2532); repair-TS-R started on GPU 1 in the same session. **All error-detection runs done.** 08:35 IST: paired bootstrap tests written (`ts/analysis/ed_significance.py`, `results/analysis/significance_ed.json`; see RESULTS.md). Remaining for error detection: the report write-up.

**2026-11-17, 08:32 IST:** TS-R seeds 1-3 finished (tmux `nlr_ts`), exit 0. Kept pairs per round: seed 1 321 / 525 / 552; seeds 2-3 in `logs/nlr_ts.log`. 08:32 IST: tmux `nlr_gen` started on GPU 0 (`TS_R_GPUS=0`, log `logs/nlr_eval.log`): R test generation for vanilla and the 9 TS-R models, both prompts. Scoring is not chained: `r_eval score` holds the CPU validation slot for its whole run, which would stall repair-TS-R's validation, so it is scheduled once the program count is known. Repair-TS-R stays one process on GPU 1, seeds 1-3 in order (seed 1 in round 3); it is not split across GPUs, because a second process on seed 3 could reach the same run directory as the first.

**2026-11-17, 12:46 IST:** `nlr_gen` finished, exit 0: 80 cells (vanilla + 9 TS-R models x 8), 7,122 cases per model, about 99.8% with R code. Repair-TS-R seed 1 done (recovered 101 / 84 / 102 on 321 / 533 / 579 kept); seed 2 round 1 done. **12:50 IST: repair-TS-R split across GPUs** to save about 8 h: `edsd3` stopped during seed 2 round 2's repair generation (cached LLM responses; no training running, so no checkpoint in play; Ctrl-C left the process waiting on worker threads, so it and its vLLM server were sent SIGTERM; both GPUs confirmed empty). Relaunched as tmux `nlr_rep2` (seed 2 only, GPU 1, log `logs/nlr_repairts.log`) and `nlr_rep3` (seed 3, GPU 0, log `logs/nlr_repairts3.log`), both `TS_R_HOLD_GPU=1`, so no two processes share a seed. tmux `nlr_score` (CPU) scores the 80 cells: 39,668 unique R programs (log `logs/nlr_eval.log`). Repair-TS-R seeds 2-3 expected about 20:00-22:00 IST; then generation and scoring of the 9 repair-TS-R models.

**2026-11-17, 18:22 IST:** repair-TS-R seed 2 finished (tmux `nlr_rep2`), exit 0: kept + recovered by round 352+88 / 525+117 / 564+126. Seed 3 (tmux `nlr_rep3`, GPU 0) in round 3 (round 1 336+90, round 2 499+117). TS-R test scoring finished 13:49 IST, exit 0 (80 cells; RESULTS.md, Phase 9). 18:23 IST: tmux `nlr_gen2` on GPU 1 generates R test answers for repair-TS-R seeds 1-2 (6 models, log `logs/nlr_eval.log`). Scoring of the 9 repair-TS-R models waits until seed 3 has finished, so that it never holds the CPU validation slot while seed 3 needs it.

**2026-11-17, 20:54 IST:** repair-TS-R seed 3 finished (tmux `nlr_rep3`), exit 0. All NL-to-R training done. 20:55 IST: tmux `nlr_gen3` (GPU 0): test generation for repair-TS-R seed 3, then (after `nlr_gen2` closes) scoring of all 9 repair-TS-R models, log `logs/nlr_eval.log`.

**2026-11-17, 22:46 IST: NL-to-R complete.** `nlr_gen3` finished, exit 0 (seed 3 generation + scoring of the 72 repair-TS-R cells). Significance: repair-TS-R beats TS-R at every round, all seeds, p < 0.0001 (RESULTS.md, Phase 9). Both GPUs idle. Open: report rebuild (`report/build.sh`) and text update for error detection and NL-to-R; optional R same-data control.

**2026-11-18: LaTeX report rebuilt with the final results** (`report/main.pdf`, 127 pages; 0 errors, 0 undefined references; 24 of 24 cross-checks match). Generator (`report/tools/build_tables.py`, backup `backups/build_tables.py.before_final_20261118`) now also writes `nlr`, `nlr_stats`, `ed_summary`, `significance_r`, `significance_ed` (38 files). Text updated: abstract, title page, methodology (NL-to-R), history (Phase 7, timeline to 17 Nov), results ch. 9 (repair-ED with controls; NL-to-R section), failures F17 (R feedback) and F18 (GPU taken), comparison table (NL-to-R row), discussion, limitations, conclusion (contributions, future work), appendices C (two new significance tables) and H (R commands). Backups of the chapters and appendices: `backups/report_{chapters,appendices}_before_final_20261118/`. Remaining overfull boxes are small (under 8 pt, one of 47 pt in the repair-flow figure) and predate this update.

- [x] 2026-11-18 11:01 IST — Report wording fixes after re-checking the repair slide numbers (backups in `backups/report_before_wording_20261118/`):
  - "twelve days" → fourteen (dated records 2026-11-04 to 2026-11-17) in the abstract, §5.3 and Table 12.3.
  - Repair-ED: "gone by round 3" replaced in the abstract and conclusion (list and Table 15.1). It stays ahead at rounds 2–3 (+0.006, +0.011) but not significantly (p 0.70, 0.58), with seeds split.
  - Table 13.1: stale error-detection row ("1 seed: mixed") updated to 3 seeds; NL-to-R row added; Effect column widened.
  - NL-to-R recovery corrected from "20–30%" to 16–31% (25–31% in round 1, 16–23% in rounds 2–3, from runs/train_rrepairts_instruct_seed*/round*/summary.json) in §9.3 and RESULTS.md.
  - Rebuilt report/main.pdf: 127 pages, 0 errors, 0 undefined references, cross-check 24/24.

- [x] 2026-11-18 11:29 IST: End-semester slides built in LaTeX (Beamer), matching the mid-semester deck's design: `presentation/endsem.tex` with theme `presentation/beamerthemetsdeck.sty`. Build with `presentation/build.sh` (same pinned TeX Live image as the report) to get `presentation/endsem.pdf`.
  - 35 pages: title, roadmap, 5 section dividers, 20 numbered slides, thank-you page, and 7 backup slides numbered separately.
  - Follows docs/PRESENTATION_PLAN.md (Draft 2). Added a "mid-semester plan vs. what we ran" slide and a round-3 confidence-interval slide.
  - Chart colours checked for colour-blind separation (worst pair ΔE 12.8, OKLab ×100). Every slide was rendered and checked by eye.
  - The presentation date is a placeholder (\tsdate = November 2026).

- [x] 2026-11-18 11:50 IST: Slides revised after user feedback (previous version: `backups/endsem.tex.before_pertask_20261118`). Now 41 pages, with 26 numbered main slides.
  - Added a chart slide per task: NL-to-Scala bars vs. paper; error-detection bars vs. paper; an "all four tasks" summary chart in the style of the mid-semester GPT chart.
  - Added repair-by-round charts for base NL-to-SQL, NL-to-Scala and error detection, next to NL-to-R; added a test-time voting chart to the main deck.
  - Removed dates and durations: "14 days" replaced by "4 tasks"; the backup timeline is now an undated "How the work progressed" flow.
  - The user asked for a faked longer run time from mid-sem. I declined and did not invent any dates or durations.
  - Negative results kept as they were, at the user's choice.

- [ ] 2026-11-18 12:20 IST: Backup of the whole project (123 GB) to E:\table-specialist-repro-backup\ (8 TB HDD, a separate physical disk from the NVMe drives). Script `scripts/backup_to_windows.sh`, tmux session `backup_win`, log `logs/backup_windows_20261118.log`.
  - Exact tar archive with SHA-256 and a file list, verified with `tar --compare`.
  - Browsable folder copy for Windows, without .venv, verified with an rsync dry run.
  - README.txt explains how to restore.
  - Smoke test passed: tar round-trip identical; MLflow files with ':' in their names copy correctly.
  - 15:09 IST: steps 1–3 done (archive 132 GB, 309,933 entries; SHA-256; file list). `tar --compare` flagged only files edited during the backup (presentation sources and build outputs, the backup log), so the archive is an exact 12:20 IST snapshot; it holds the 41-page deck. Step 4 (folder copy) started 15:05 IST.

- [x] 2026-11-18 15:09 IST: Slides rebuilt for a 30-minute slot after user feedback (previous version: `backups/endsem.tex.before_30min_20261118`, theme backup `backups/beamerthemetsdeck.sty.before_30min_20261118`). Now 59 PDF pages: 43 numbered main slides, 6 section dividers, title, thank-you, backup divider and 7 backup slides. 0 errors; every page rendered and checked by eye.
  - New section 1, "The paper": problem, problem statement, the key idea (generative tasks vs. classification tasks), the full architecture diagram, a worked execution-validation example, a worked permutation-validation example, why each part exists, and the paper's reported results.
  - Removed: "mid-semester plan vs. what we ran", "what we built and how far you can trust it", "repair never hurts", "round 3: three clear gains", "test-time voting" (voting stays in backup).
  - Plan slide redone as an experiment grid plus five stages. "Filling the gaps" now has four slides: expected vs. found, what the code does not do, the vanilla gate (base vs. Instruct), the comparator and its audit.
  - Reproduction: setup slide; an NL-to-R per-test-set chart added. The two vanilla bars are recoloured light red (ours) and steel blue (paper); worst pair still ΔE 12.8 under colour-blind simulation.
  - Idea 1 split into best-of-4, DPO, DPO dev tuning, and a verdict-and-lessons slide. Idea 2: repair drawn inside the full TS architecture, plus detail, pilot, and a recovered-data chart.
  - Results: scoreboard table (TS vs. repair, every setting and round), then one slide per task with a round-3 per-test-set table and a zoomed rounds chart.
  - "Is it just more data?" split into three slides: the test design, the round-1 decomposition for base SQL, Instruct SQL and error detection (with best-of-4 and best-of-2), and what the controls show.
  - Numbers from results/analysis/*.json, results_table.md, report/generated/*.tex and runs/train_rrepairts_instruct_seed*/round*/summary.json.
  - Found while doing this: report text says repair-ED recovered "about 450–650 errors per round"; report/generated/ed_rounds.tex gives 491–696. The slides use 491–696; the report is not yet corrected.


- [x] 2026-11-18 15:20 IST: NL-to-R reproduction slide now shows the paper's bars, as the other tasks do. The paper reports only an NL-to-R average over 5 sets (0.260 → 0.333, unverified, includes Text2Analysis), so the paper bars sit in a separate "Average" panel next to our zero-shot mean of 4 (0.327 → 0.476) and are labelled as not like-for-like. The per-test-set panel says the paper does not report them. No per-set paper values were invented. Backup: `backups/endsem.tex.before_rpaperbars_20261118`.

- [x] 2026-11-18 15:24 IST: Correction. The user supplied the published paper PDF (EMNLP 2025, pp. 35443–35460). Its Table 3 **does** give NL-to-R per test set for Llama-3.1-8B: WikiSQL 0.331→0.409, WikiTQ 0.138→0.257, Text2Analysis 0.103→0.199, Spider 0.503→0.536, BIRD 0.225→0.261 (average 0.260→0.333). My earlier claim that the paper reports only the average was wrong; it came from the summary-only reading of the PDF. Mean of the 4 public sets: 0.299→0.366 (ours 0.327→0.476).
  - The PDF also confirms the NL-to-Scala (0.359/0.728, 0.043/0.188, 0.249/0.356, 0.100/0.228) and error-detection (ST 0.071/0.136, RT 0.108/0.161) figures used before, so they are no longer 'unverified'.
  - Slides: NL-to-R slide now has paper vanilla and paper TS bars per test set, like the other tasks; paper-results table has the full NL-to-R row; summary chart has a paper NL-to-R point; every 'unverified' dagger is removed. Backup: `backups/endsem.tex.before_paperpdf_20261118`.
  - Not yet done: the report (ch. 2, 9, 12, 14) and RESULTS.md still call these figures unverified and the NL-to-R figure 'not comparable'.

- [x] 2026-11-18 15:32 IST: Report corrected against the published paper and rebuilt (`report/main.pdf`, 127 pages, 0 errors, 0 undefined references; tables regenerated, 38 files). Backups: `backups/report_before_paperpdf_20261118/` (chapters, appendices, tools, references.bib, RESULTS.md).
  - New bibliography entry `xing2025emnlp` (EMNLP 2025, pp. 35443–35460). All 'unverified' markers removed (chapters 0, 1, 2, 6, 9, 12, 14, 15 and the error-detection summary note in `tools/build_tables.py`); none remain in the PDF.
  - Ch. 2: tasks table (validation, metrics) and results table updated from published Tables 2–3, with NL-to-R per set and Text2Analysis values. The 'two tables disagree' and 'average does not match its parts' points replaced: the ablation reports round 1, and the paper's averages include Text2Analysis. Added: Table 3's GPT-3.5 values equal the last iteration in Figure 6.
  - Ch. 9 and 12: NL-to-R compared per set with the paper (TS above on all 4 sets, +0.110 on the mean of 4; vanilla within 0.012 on 3 of 4). Ch. 14 and 15, abstract updated to match.
  - Also fixed the two errors flagged earlier: base TS keeps 13.4–26.7% (was 13.4–25.1%, ch. 7); repair-ED recovered 491–696 errors per round (was 450–650, ch. 9 and 13).
  - docs/RESULTS.md: dated correction note added under 'Other tasks in the paper'; the Scala 'figures disagree' note and the NL-to-R paper row corrected.

- [x] 2026-11-18 15:45 IST: Speaker transcript written: `docs/PRESENTATION_TRANSCRIPT.md`. Per slide (title, 1–43, backups): time budget (30 min total), full script, must-say lines, what to point at, what not to say, transition. Also: the ten must-say sentences, a suggested speaker split (Fadil 1–15, Tajamul 16–43), time checkpoints, a backup-slide guide, 25 likely questions with answers, and a delivery checklist. Review against the deck and the published paper found six possible slide fixes (listed in the reply to the user; not applied).
