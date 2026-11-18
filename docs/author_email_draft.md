CANCELLED 2026-11-04: contacting the authors is not possible. Kept only as a list of what stays unresolved.

To: junjiexing@microsoft.com
Subject: Questions on reproducing Table-Specialist (EMNLP 2025)

Hello,

I am a student at IIT Jammu reproducing the Llama-3.1-8B NL-to-SQL results from your Table-Specialist paper for a course project. The repository has been very helpful. I have rebuilt the test sets and evaluated the vanilla model, and a few details aren't stated in the paper or repository. Anything you can share would help:

1. Which Llama-3.1-8B checkpoint did you use: base (`meta-llama/Llama-3.1-8B`, which your README links) or Instruct? Did you call it through chat or raw completions?

2. Which prompt produced the vanilla scores in Table 3: the zero-shot test prompt from `Finetune/main_build_data.py`, or the four-example prompt in Appendix A? On base Llama we get these scores, against your vanilla scores:

   | Benchmark | Zero-shot | Four-example | Your paper |
   |---|---|---|---|
   | WikiSQL | 0.326 | 0.685 | 0.525 |
   | Spider | 0.725 | 0.769 | 0.670 |
   | WikiTQ | 0.219 | 0.344 | 0.300 |
   | BIRD | 0.364 | 0.399 | 0.388 |

3. For WikiSQL, were the table cells in the prompt lowercased? `ns_evaluator.py` lowercases the gold SQL, so we assumed lowercase data in the database and in the prompt.

4. For the test sets, could you share the case ID lists?
   - Spider: we took the test split, kept gold SQL touching one table (1,230), and applied the 70% prompt limit, which gives exactly your 1,198. Is this how you built it?
   - BIRD: the same filter on dev gives 363, not 356. What else did you remove?
   - WikiSQL and WikiTQ: how were the 1,000 cases sampled?

5. Which table corpus did you use, and how many tables per round for the main results? Your script names suggest web tables. We are using GitTables because the corpus isn't released.

6. What LoRA rank, number of epochs and learning rate did you use for Llama, what N for row-sample validation, and what sampling temperature?

7. In the Llama NL-to-SQL runs, what share of generated pairs passed execution validation, and how many training examples did each round have? We keep 41 of 300 tables (13.7%) in round 1 with Llama writing the questions, SQL and Spark-Scala. Which model generated the code and questions in those runs?

8. How did you compare SQL and Scala results? The validation script runs both but has no comparison step. In particular, did a pair with the right rows but extra output columns count as agreeing, and how did you treat execution errors and empty results?

Even a partial answer would help a lot.

Thank you,
[your name]
