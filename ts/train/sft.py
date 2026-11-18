"""LoRA fine-tuning with the paper's and the config's settings. Completion-only loss, so prompts cost no gradient.

Each round starts from vanilla weights and trains only on that round's examples (paper Algorithm 1 resets the training
sets each round, and Section 6.5 finds vanilla better than continuing from the last checkpoint).

Run as a script so that it owns one GPU and frees it on exit:
  python -m ts.train.sft --model models/Llama-3.1-8B --examples <file.jsonl> --output <dir> --seed 1
"""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import yaml

from ts.repo import ROOT


def build_example(generator, table, question, code):
    """Format one training example with the repo's own classes. Returns None if the prompt is over the repo's limit."""
    prompt = generator.generate_prompt((table, question, None), None)
    if prompt is None:
        return None
    return {"prompt": prompt, "completion": generator.generate_completion(code)}


CHECKPOINT_FRACTION = 1 / 12  # about 12 checkpoints per run on small adapters
CHECKPOINT_SECONDS = 300  # large adapters: at most about this much training lost to a crash
SECONDS_PER_EXAMPLE = 1.2  # measured: 1.16-1.26 s per example across both arms of seed 1


def accumulation_steps(example_count, batch_fraction):
    """Paper: a batch is 1% of the training data, so that each epoch has 100 steps. One example fits on the GPU at a
    time, so the batch is built by accumulating gradients."""
    return max(1, round(example_count * batch_fraction))


def checkpoint_steps(example_count, accumulation, epochs, max_steps=None):
    """Steps between checkpoints: 1/12 of the run, or about 5 minutes of training if that is shorter."""
    total = max_steps or math.ceil(example_count / accumulation) * epochs
    return max(1, min(math.ceil(total * CHECKPOINT_FRACTION), round(CHECKPOINT_SECONDS / (accumulation * SECONDS_PER_EXAMPLE))))


def load_examples(path):
    # Only "\n" ends a row: str.splitlines also breaks on characters such as U+2028, which table text can contain.
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def train(model_path, examples, output_dir, seed, settings, max_steps=None):
    # Imported here so that the rest of the pipeline does not need a GPU stack just to import this module.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 0}, dtype=torch.bfloat16)

    accumulation = accumulation_steps(len(examples), settings["batch_fraction_of_data"])
    save_steps = checkpoint_steps(len(examples), accumulation, settings["epochs"], max_steps)

    class FlushCheckpoints(TrainerCallback):
        def on_train_begin(self, _args, state, _control, **_kwargs):
            state.save_steps = save_steps  # a resumed run would otherwise keep the interval saved in its checkpoint

        def on_save(self, *_args, **_kwargs):
            os.sync()  # a checkpoint that only reached memory would not survive a power cut

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list(examples),
        peft_config=LoraConfig(r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
                               target_modules=settings["target_modules"], task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=str(output_dir), num_train_epochs=settings["epochs"], max_steps=max_steps or -1,
            per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
            gradient_checkpointing=True, learning_rate=settings["learning_rate"],
            lr_scheduler_type=settings["scheduler"], bf16=True, max_length=settings["max_seq_length"],
            completion_only_loss=True, logging_steps=10, report_to="none", seed=seed,
            save_strategy="steps", save_steps=save_steps, save_total_limit=1),
        callbacks=[FlushCheckpoints()],
    )
    resume_from = get_last_checkpoint(str(output_dir)) if Path(output_dir).is_dir() else None
    if resume_from:
        print(f"Resuming from {resume_from}", flush=True)
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.model.save_pretrained(output_dir)
    os.sync()  # the adapter must be on disk before the summary that marks it finished
    summary = {"examples": len(examples), "accumulation_steps": accumulation, "global_steps": result.global_step,
               "train_loss": round(result.training_loss, 4), "seconds": round(time.perf_counter() - start, 1),
               "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1), "seed": seed,
               "resumed_from": resume_from and Path(resume_from).name}
    temporary = Path(output_dir) / "train_summary.tmp"
    temporary.write_text(json.dumps(summary, indent=1))
    temporary.rename(Path(output_dir) / "train_summary.json")
    os.sync()
    for checkpoint in Path(output_dir).glob("checkpoint-*"):
        shutil.rmtree(checkpoint)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="Only for short tests")
    args = ap.parse_args()
    settings = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())["training"]
    print("RESULT " + json.dumps(train(args.model, load_examples(args.examples), args.output, args.seed, settings,
                                       args.max_steps)))


if __name__ == "__main__":
    main()
