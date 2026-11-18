# Presentation plan: Table-Specialist reproduction

Draft 2, 2026-11-18. Built as `presentation/endsem.tex` (see docs/TASKS.md, 2026-11-18). **Superseded at 15:09 IST** by the 30-minute version: the deck now has a six-section structure (paper, plan, gaps, reproduction, improvement, conclusions); see the TASKS.md entry of that time for what changed. This file is kept as the record of Draft 2. Slide 14 has been corrected against `results/analysis/significance*.json`, and slides 15–19 have been updated to match it.

## Assumptions and framing

- **Audience and length.** Faculty evaluators who know ML but not this paper. The talk runs 15–20 minutes: about 19 main slides plus backup slides.
- **Through-line.** The paper's validator throws away 75–89% of the data it generates. The talk follows that waste in three acts:
  1. Act 1, reproduce: can we get the paper's numbers?
  2. Act 2, understand: what does validation buy you?
  3. Act 3, improve: can we recover the rejected data?
- **Slide titles state claims.** Someone reading only the titles should get the whole argument.
- **Colours and labels.** Keep the report's colours: paper = amber, our TS = navy, repair = sea green, vanilla = grey. Mark the paper's non-SQL numbers as unverified.

## Opening (about 3 min)

### 1. Title

- Title, name, course and date.
- Subtitle: "Reproducing and extending Table-Specialist (EMNLP 2025) on Llama-3.1-8B".

### 2. Fine-tuning a table specialist without human labels

- Generic LLMs are weak at table tasks, and labelled data is expensive.
- The paper's idea: pair each task with a dual task that can check it. For NL-to-SQL, a Spark-Scala program for the same question must return the same rows.
- *Visual:* one SQL/Scala pair agreeing on a small table.

### 3. How the loop works

- Generate questions, then SQL and Scala.
- Validate on 10 random half-samples of the table.
- Train with LoRA on agreeing pairs only, and repeat for 3 rounds.
- One line on error detection: an injected error must be found by the validator.
- *Visual:* `report/figures/paper-loop.tex`.

### 4. Our two goals

- Goal A: match the paper's Llama-3.1-8B numbers.
- Goal B: improve on them.
- Hook: "The validator kept only 11–25% of pairs. Can we use the rest?"

## Act 1: Reproduction (about 5 min)

### 5. What the paper doesn't tell you

- **Not released:** the training corpus, the Text2Analysis test set, the test-case ID lists.
- **Not stated:** base or Instruct checkpoint, LoRA settings, which round is reported, the test prompt.
- **Not in the code:** the validator never compares SQL with Scala output.
- **What we did about it:** a GitTables corpus of 10,053 tables, test sets rebuilt from the original datasets, and our own comparator, audited by hand.
- Message: reproduction was detective work.

### 6. What we built and how far you can trust it

- Pipeline diagram (`report/figures/our-pipeline.tex`).
- 2× RTX 4500 Ada GPUs, 14 days, 3 seeds × 3 rounds per arm.
- Paired bootstrap tests, with the sign of each seed's difference reported.
- Resumes after a crash: it survived a crash, a 7.5-hour stall and a WSL crash with no lost results.
- 205 tests; generated code runs only in Docker sandboxes.

### 7. First surprise: the README's checkpoint can't match the paper

- Base Llama failed the vanilla check under every setting tried (WikiSQL 0.326 against 0.525).
- Instruct, queried through the repository's own chat call, comes within 0.015 on 3 of 4 test sets.
- So the paper's numbers most likely come from Instruct.

### 8. Headline: Instruct TS matches the paper

- *Visual:* per-test-set bars for our vanilla, paper vanilla, our TS and paper TS (report §7.4).
- Zero-shot mean of 4: 0.6113 ± 0.0135, against the paper's 0.5983.
- Per-set differences: WikiSQL −0.032, WikiTQ −0.026, BIRD −0.006.
- Spider is +0.116 for every model, vanilla included, so it is a test-set offset.

### 9. Validation only pays off over rounds

- *Visual:* rounds line chart, base TS against NV (no validation).
- Round 1: NV wins 17 of 24 comparisons, because it trains on 6.6× more data.
- Round 3: TS wins 20 of 24. It beats NV by +0.049 and vanilla by +0.065 (p < 0.0001, all seeds).
- NV falls every round, so errors compound without validation.
- The paper does not show this round-1 crossover.

### 10. The other tasks reproduce too

| Task | Our vanilla / paper vanilla | Our TS / paper TS |
|---|---|---|
| NL-to-Scala (zero-shot mean of 4) | 0.194 / 0.188† | 0.467 / 0.375† |
| Error detection (F1; ST shown) | 0.066 / 0.071† | 0.279 / 0.136† |
| NL-to-R (zero-shot mean of 4) | 0.327 | 0.476 (+0.128 over vanilla on mean of 8) |

† Paper figure unverified. NL-to-R uses a validator we wrote, because the repository has none. The paper's R figure averages 5 test sets, so it is not comparable.

## Act 3: Improving (about 6 min)

### 11. The waste

- Validation outcomes: 11–25% of pairs kept.
- Rejection reasons: disagree, scala_failed, always_empty. On Instruct, about 1,000 pairs per round are always_empty.
- Two ideas: learn from rejected pairs, or repair them.

### 12. Idea 1 failed: preference learning

- DPO on validated against rejected pairs: −0.0209 (p < 0.0001), even after tuning on the dev sets.
- Best-of-4 sampling gives a small, unstable gain.
- Lesson: use the validator to select data, not to define negatives.

### 13. Idea 2: repair-TS

- *Visual:* `report/figures/repair-flow.tex`.
- Each rejected pair gets one repair, using its own execution feedback. It is then re-checked by the paper's unchanged validator.
- The go/no-go rule was fixed before the pilot.
  - Base as its own repairer: +17% recovered, so no-go.
  - Instruct as repairer: +56% recovered, and a hand check found 21 of 23 recovered pairs correct.

### 14. Repair never hurts, and helps clearly in 3 of 5 settings (corrected)

Every number is repair minus plain TS at the same round, on the same seeds (1–3) and the same test cases.
- **SQL, Scala and R:** "mean of 8", which is 4 test sets × 2 prompts (zero-shot and few-shot), averaged over 3 seeds.
- **Error detection:** F1, averaged over the ST and RT test sets.
- **p-values:** paired bootstrap over test cases (over test columns for error detection), 10,000 draws.

| Setting | Round 1 | Round 2 | Round 3 | Round 3, per seed | Verdict |
|---|---|---|---|---|---|
| Base NL-to-SQL | +0.024 | +0.043 | **+0.042** | +0.020, +0.056, +0.050 | **Clear** (p < 0.0001) |
| Instruct NL-to-Scala | +0.014 | +0.034 | **+0.017** | all positive | **Clear** (p < 0.0001) |
| Instruct NL-to-R | +0.011 | +0.020 | **+0.024** | all positive | **Clear** (p < 0.0001) |
| Instruct NL-to-SQL | −0.004 | +0.005 | **−0.001** | −0.004, +0.005, −0.005 | Within noise (p = 0.55) |
| Error detection (F1) | **+0.027** | +0.006 | **+0.011** | −0.032, +0.028, +0.037 | Round 1 only (p = 0.03); then p = 0.70, 0.58 |

- **Visual:** a dot-and-interval plot of the round-3 difference against a zero line.
  - Base SQL: [+0.037, +0.048]
  - Scala: [+0.012, +0.022]
  - R: [+0.019, +0.030]
  - Instruct SQL: [−0.006, +0.003]
  - Error detection: [−0.026, +0.050], in its own panel because F1 is on a different scale.
- **Say aloud:**
  - On average, repair is ahead in 13 of 15 (setting, round) comparisons; the two exceptions are tiny Instruct-SQL differences.
  - In three settings the gain is clear on every seed.
  - In the other two it is too small to separate from noise.
- **Do not say** "repair helps everywhere", and do not say the error-detection gain "disappears". It is still +0.011 at round 3, but not significant.

### 15. Is it just more data?

- **Base SQL, same-data control at round 1:** about half the gain is data volume.
  - Control vs TS: +0.012.
  - Repair vs control: +0.012, positive on every seed.
- **Instruct SQL:** extra fresh pairs help (+0.008), but repaired pairs are worth less than fresh ones (repair vs control −0.011, p < 0.0001).
- **Error detection, round 1:** repair beats the same-data control (+0.023, p = 0.03), but not a plain second sample (best-of-2: +0.006, p = 0.71).
- **Hypothesis, not tested:** repair helps where first-pass data is weakest: base SQL, Scala and R. It does not help where data is already strong (Instruct SQL).
- **Caveat:** on base, the repairer is a stronger model, so part of the gain may be distillation.

### 16. NL-to-R: our cleanest new result

- *Visual:* zero-shot mean of 4 by round.
  - TS-R: 0.434, 0.458, 0.476.
  - Repair-TS-R: 0.444, 0.481, 0.504.
  - Vanilla: 0.327 (a flat line).
- A new pipeline end to end, scored on 7,122 test cases per model.
- Repair recovered 16–31% more pairs: 25–31% in round 1 and 16–23% after.
- Every seed is positive at every round.
- Flag it: no same-data control for R, so how much of the gain is volume is not measured.

## Close (about 2 min)

### 17. What's established and how strongly

A condensed Table 15.1 from the report:

| Claim | Strength |
|---|---|
| Repair-TS on base SQL, Scala and R | Moderate |
| Repair-TS does not help Instruct SQL | Strong (negative) |
| Repair-ED | Weak: not significant beyond round 1 |
| Round-1 crossover | Moderate |
| DPO harms | Moderate |

### 18. Limitations and future work

- **Limitations:**
  - substitute corpus;
  - the paper's non-SQL figures are unverified;
  - 3 seeds only;
  - no R same-data control;
  - small error-detection test sets (42 and 36 columns with errors).
- **Next:** the R same-data control, separating repair from distillation, and repairers from other model families.

### 19. Three takeaways

1. The paper reproduces, but on Instruct, not the base checkpoint its README points to.
2. Validation's value shows over rounds, not in round 1.
3. Repairing rejected data never hurt. It clearly helps where first-pass data is weak (base SQL, Scala, R), and controls tell you how much is just more data.

## Backup slides

- Per-test-set and per-seed tables.
- The comparator defects found by the hand audit.
- Self-consistency: +0.014 to +0.025 everywhere.
- Few-shot vanilla ≈ zero-shot TS.
- Full error-detection table with significance tests.
- **B-14 (new): repair per round with per-seed differences for all five settings.** It has the zero-shot-only view, where Instruct SQL is slightly positive at rounds 2–3 (+0.0065, +0.0009) because few-shot is what pulls the mean of 8 negative.
- Incident timeline, hyper-parameters, compute cost, bootstrap method.

## Questions to prepare for

- **"Why Instruct when the README says base?"** See slide 7.
- **"Agreement isn't correctness."** Hand audit: 39 of 50 kept pairs fit their question.
- **"Is repair just distillation, or just more data?"** Partly; see slide 15. Say so plainly.
- **"Your table shows repair higher on Instruct SQL zero-shot. Why call it no effect?"** It is +0.001 at round 3 on zero-shot and −0.001 on the mean of 8. The seeds split, and the interval crosses zero.
- **"Why are you above the paper on Scala?"** It can't be attributed: corpus, checkpoint and round choice are all candidates.
- **"Do the p-values cover seed variance?"** No. That's why each seed's sign is reported alongside.
