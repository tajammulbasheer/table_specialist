# Plan: match the paper, then improve on it (drafted 2026-11-10)

Status: **proposal, waiting for the user's approval.** Nothing here has been run yet.

Two goals:

- **Goal A: match the paper's Llama-3.1-8B NL-to-SQL numbers**, using the paper's models and datasets as far as they are available.
- **Goal B: improve on them** with established methods from the literature, not new ideas of our own.

## 1. What the analysis found

### 1.1 The paper (arXiv 2410.12164, HTML v2, dated 23 Mar 2026)

Target, Table 3, Llama3.1-8B NL-to-SQL, vanilla → Table-Specialist:

| Test set | WikiSQL | WikiTQ | Text2Analysis | Spider | BIRD | Average of 5 |
|---|---|---|---|---|---|---|
| Vanilla | 0.525 | 0.300 | 0.273 | 0.670 | 0.388 | 0.431 |
| Table-Specialist | 0.816 | 0.449 | 0.465 | 0.690 | 0.438 | 0.572 |

Facts that matter for matching, all checked in the HTML:

- **Model:** "Llama-3.1-8b". The paper never says base or Instruct.
- **Test prompt:** not stated. The paper does not say whether it is zero-shot or few-shot, chat or completion, or whether a system message is used.
- **Fine-tuning:**
  - LoRA, with "learning-rate multiplier of 0.5" (an OpenAI term) and batch size at 1% of the data.
  - Rank, epochs and absolute learning rate are not stated.
- **Rounds:**
  - "up to 3".
  - The round each Table 3 cell comes from is not stated.
  - Algorithm 1 reads as continued fine-tuning, while Section 6.5 says restarting from vanilla is "consistently better". Which one Table 3 uses is not stated.
- **Training data:**
  - The corpus is described only as "a large corpus" of "real tables". The repo's file names say web tables (`table2question_all_webtable_v4_1st_3K`).
  - 3,000 is given only as an example step size.
  - No pass rate and no count of kept pairs are given.
- **Test sets:** sizes as ours, except BIRD 356 (ours 363). Splits are not stated. Text2Analysis is not public.
- **No-validation:** shown only in a figure (no numbers). The only numeric ablation, Table 11, is GPT-3.5.
- **The ablation tables disagree with Table 3** for the same setting: NL-to-SQL is 0.594 in Table 11 but 0.616 in Table 3. So the paper's own numbers vary by about ±0.02 between runs.

### 1.2 The authors' code (`third_party/Table-Specialist`, commit `af1921f`)

- **Evaluation call:** `Evaluete/queryWithDeployment.py` queries only through **chat completions with a fixed system message**: "You are an AI assistant. Provide an answer to some data/table related questions."
  - There is no completion (non-chat) path.
  - A base model has no chat template, so the published code cannot query base Llama as written.
- **README:** links `meta-llama/Llama-3.1-8B`, the base checkpoint. This is the only evidence for base.
- **Test prompts (`Finetune/main_build_data.py`):** zero-shot by default.
  - Few-shot test prompts appear only when `--sample_method random-Gen` or `random-Ben` is passed.
  - The evaluator scores each prompt type separately (`method = sampleMethod + numSamples`).
- **Training prompts:**
  - The default settings (`num_few_shot_samples=10`, `sample_method="random"`) raise an error in `generate_train_one_dataset`.
  - So the authors must have trained either with `random-Gen` (each training prompt carries examples drawn from the generated data) or with no few-shot examples. This is not recorded.
  - **Ours are zero-shot only.**

### 1.3 Our own earlier results that point to the cause of the vanilla gap

| Model and request | WikiSQL | WikiTQ | Spider | BIRD |
|---|---|---|---|---|
| Paper vanilla | 0.525 | 0.300 | 0.670 | 0.388 |
| Base, completions, zero-shot (the faithful setting used so far) | 0.326 | 0.219 | 0.7245 | 0.3636 |
| **Instruct, chat, repo system message, zero-shot** | **0.518** | **0.289** | 0.7913 | not run |

- **Instruct through the repo's own chat call** lands within 0.011 of the paper on WikiSQL and WikiTQ.
- **Base misses** by 0.199 and 0.081.
- **Spider is above the paper with every model**, base too. The paper's own GPT-3.5 Spider score (0.650) is also unusually low. Our Spider case set, or the paper's, differs, and this cannot be resolved without the authors' case list.

**Conclusion:** the strongest available explanation is that the paper's Llama numbers come from **Llama-3.1-8B-Instruct queried through the repo's chat call**, not from base. This is a hypothesis, not a fact; the README link argues for base.

### 1.4 Where our reproduction stands (3 seeds, round 3, zero-shot)

| Approach | WikiSQL | Spider | WikiTQ | BIRD | Mean of 4 |
|---|---|---|---|---|---|
| Paper TS | 0.816 | 0.690 | 0.449 | 0.438 | 0.5983 |
| Our TS, base | 0.7790 | 0.7031 | 0.3217 | 0.3802 | 0.5460 |
| Our repair-TS, base | 0.7647 | 0.7843 | 0.3947 | 0.4178 | 0.5904 |

## 2. Goal A: experiments to match the paper

Ordered by how much each could explain, and how cheap it is.

| # | Experiment | Why | Source of the setting | Cost | Decision rule |
|---|---|---|---|---|---|
| A1 | **Instruct vanilla gate:** all 4 test sets, chat with the repo system message, zero-shot and few-shot; BIRD for the first time | Section 1.3: the likeliest cause of the vanilla gap | Repo (chat call, system message) | About 1 h, one GPU | If Instruct is within ±0.05 on WikiSQL, WikiTQ and BIRD, go to A2 |
| A2 | **Instruct TS, 3 seeds × 3 rounds:** the same loop, with generation, training and evaluation all in Instruct's chat format with the repo system message | Matches the paper's Llama numbers if A1 holds | Paper loop, repo prompts and call | About 30 h (CPU validation is the limit, as before) | Compare with Table 3, round by round |
| A3 | **Instruct no-validation, seed 1 only** | The paper's ablation on the matched model; the full 3 seeds would add about 70 h of GPU | Paper Section 6.1 | About 24 h of GPU, alongside A2 | TS must beat NV at round 3 |
| A4 | **Training prompts with few-shot examples** (`random-Gen`, 1-5 examples from the round's own validated data), TS seed 1 | The repo cannot train with its defaults, so the authors used some setting we have not reproduced | Repo (`main_build_data.py`) | About 17 h | Keep if the dev-set mean rises; report either way |
| A5 | **Continued fine-tuning instead of restarting**, TS seed 1 | Algorithm 1 reads as continued; Section 6.5 says restarting is better | Paper | About 17 h | Report as the second reading of Algorithm 1 |
| A6 | **Web-table corpus** in place of GitTables: WDC Web Tables, English relational subset, with a contamination filter against WikiSQL and WikiTQ | The repo's file names say web tables. GitTables is more numeric and less like the test sets | Repo file names; corpus ours | Download plus about 2 days | Only if A1-A2 leave a gap above 0.05 |
| A7 | **Report the best round as well as round 3**, and BIRD with the 7 extra cases marked | The paper does not say which round it reports | Paper | Analysis only | Both are shown, labelled |

**Not possible:** Text2Analysis (not public), the paper's exact case lists, and the authors' training tables. Averages leave out Text2Analysis on both sides.

## 3. Goal B: improve the results with established methods

Each of these comes from published work and is applied to our final models. Settings are tuned on the dev sets only (WikiSQL 300, Spider 575, WikiTQ 300; built in `ts/bench/dev.py`), with the selection rule fixed before any dev result, as in the DPO tuning.

| # | Method | Source | What we would do | Cost | Expected effect |
|---|---|---|---|---|---|
| B1 | **Execution-based self-consistency** (majority vote over results) | Wang et al. 2022 (self-consistency); OmniSQL (arXiv 2503.02240) uses it for text-to-SQL | At test time, sample N SQL programs (N = 8, temperature 0.7), run them on the test table, and return a program from the largest group of identical results. Needs no training | About 1 h per model; covers vanilla, TS r3 and repair-TS r3 on 3 seeds | Literature reports gains of a few points; measured, not assumed |
| B2 | **Execution-guided filtering** | Wang et al. 2018 (execution-guided decoding) | Discard candidates that fail or return no rows before the vote (a filter inside B1) | Included in B1 | Removes the 3-4% compile/execution errors |
| B3 | **Self-debugging at test time** | Chen et al. 2023, "Teaching LLMs to Self-Debug" (arXiv 2304.05128) | When a test SQL fails to run, give the error back once and take the corrected SQL. This uses the database error only, never the gold answer | About 1 h per model | Small: errors are 3-4% of WikiSQL cases |
| B4 | **Corrective self-consistency** | CSC-SQL (arXiv 2505.13271) | When B1's top two groups are close, ask the model to choose between them | About 2 h per model | Report only if B1 helps |
| B5 | **More validated data per round** | Paper Fig. 13a (more data helps) | Raise the corpus to 12,640 tables (allowing up to 10 near-duplicate tables per header set, measured earlier), and use 4,000 tables per round | About 40 h | Tests the paper's own claim on our data |
| B6 | **Training settings tuned on dev:** epochs 2 or 3, learning rate 1e-4 or 2e-4 | Standard practice; the paper does not publish its own | Round 1 only, then the winner for 3 rounds | About 10 h | Unknown |

**Note for the write-up:** repair-TS is the training-time form of self-debugging (Chen et al. 2023), applied to rejected training pairs and filtered by the paper's own validator. It should be cited that way.

## 4. Proposed order and timing

1. **A1:** about 1 h. Decides whether A2 is worth its 30 h.
2. **B1 + B2 on the existing models:** about 7 h of GPU. Can run while A2 validates on the CPUs.
3. **A2** (+ **A3** on the other GPU): about 30 h.
4. **B1 on the Instruct models from A2**, and **repair-TS on Instruct** if A2 matches the paper: about 30 h more.
5. **A4, A5, B5, B6:** only if time remains, in that order.
6. **Phase 7:** a list explaining each difference from the paper, and the release package.

## 5. Decisions needed from the user

1. Is it acceptable to treat **Instruct** as "the paper's model" for Goal A, since the repo's evaluation call only works with a chat model? Base stays the faithful reproduction already reported.
2. Does "improve without newly proposed ideas" mean **established methods only** (section 3)? Or should new ideas from the earlier list (adapter merging, 2-of-3 language voting, mixed-prompt training) also be considered?
3. Should we spend about 24 h on A3 (Instruct no-validation)?

## Sources

- Paper: [arXiv 2410.12164](https://arxiv.org/abs/2410.12164), HTML v2; EMNLP 2025 version, [ACL Anthology 2025.emnlp-main.1795](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.1795.pdf); [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/table-specialist-language-model-specialists-for-tables-using-iterative-fine-tuning/).
- Code: [microsoft/Table-Specialist](https://github.com/microsoft/Table-Specialist) at `af1921f`: `Evaluete/queryWithDeployment.py`, `Finetune/main_build_data.py`, `Finetune/data_generator/base/base_data_generator.py`, `README.md`.
- Self-debugging: [Chen et al. 2023, arXiv 2304.05128](https://arxiv.org/abs/2304.05128v2).
- Self-consistency for text-to-SQL at scale: [OmniSQL, arXiv 2503.02240](https://arxiv.org/abs/2503.02240v1).
- Corrective self-consistency: [CSC-SQL, arXiv 2505.13271](https://arxiv.org/html/2505.13271v2).
- Execution-based similarity for SQL selection: [Query and Conquer, arXiv 2503.24364](https://arxiv.org/html/2503.24364v1).
- Our evidence: `results/baseline/final_*_instruct.json`, `final_*_base.json`, `gate_*.json`; `docs/RESULTS.md`, `docs/DECISIONS.md`.
