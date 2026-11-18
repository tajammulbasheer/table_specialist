"""DPO on top of a trained SFT LoRA adapter (extension 2; not in the paper).

The SFT adapter is loaded as trainable and TRL copies it as a frozen "ref" adapter, so the reference policy is the SFT
model, not vanilla. Reference log-probabilities are computed once before training, so the reference needs no memory
during training. Only the trained adapter is saved.

  python -m ts.train.dpo --model models/Llama-3.1-8B --sft-adapter <dir> --pairs <file.jsonl> --output <dir> --seed 1
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
from ts.train.sft import load_examples


def accumulation_steps(pair_count, batch_size):
    return max(1, min(batch_size, pair_count))


def last_logged(history, key):
    values = [entry[key] for entry in history if key in entry]
    return round(values[-1], 4) if values else None


def train(model_path, sft_adapter, pairs, output_dir, seed, settings, max_steps=None):
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 0}, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)

    accumulation = accumulation_steps(len(pairs), settings["batch_size"])
    total = max_steps or math.ceil(len(pairs) / accumulation) * settings["epochs"]
    save_steps = max(1, math.ceil(total / 12))

    class FlushCheckpoints(TrainerCallback):
        def on_save(self, *_args, **_kwargs):
            os.sync()

    rows = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    trainer = DPOTrainer(
        model=model, processing_class=tokenizer, train_dataset=Dataset.from_list(rows),
        args=DPOConfig(
            output_dir=str(output_dir), num_train_epochs=settings["epochs"], max_steps=max_steps or -1,
            per_device_train_batch_size=1, gradient_accumulation_steps=accumulation, gradient_checkpointing=True,
            learning_rate=settings["learning_rate"], lr_scheduler_type=settings["scheduler"],
            warmup_steps=settings["warmup_ratio"], bf16=True,  # transformers 5: a float below 1 is a ratio beta=settings["beta"],
            loss_type=settings["loss_type"], loss_weights=settings["loss_weights"],
            max_length=settings["max_seq_length"], precompute_ref_log_probs=True, logging_steps=5,
            report_to="none", seed=seed, save_strategy="steps", save_steps=save_steps, save_total_limit=1),
        callbacks=[FlushCheckpoints()],
    )
    resume_from = get_last_checkpoint(str(output_dir)) if Path(output_dir).is_dir() else None
    if resume_from:
        print(f"Resuming from {resume_from}", flush=True)
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.model.save_pretrained(output_dir, selected_adapters=["default"])
    os.sync()
    history = trainer.state.log_history
    summary = {"pairs": len(pairs), "accumulation_steps": accumulation, "global_steps": result.global_step,
               "train_loss": round(result.training_loss, 4),
               "last_reward_accuracy": last_logged(history, "rewards/accuracies"),
               "last_reward_margin": last_logged(history, "rewards/margins"),
               "seconds": round(time.perf_counter() - start, 1),
               "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1), "seed": seed,
               "resumed_from": resume_from and Path(resume_from).name, "settings": settings}
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
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="Only for short tests")
    ap.add_argument("--overrides", default="{}", help="JSON settings that replace configs/base.yaml extension2.dpo")
    args = ap.parse_args()
    settings = {**yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())["extension2"]["dpo"],
                **json.loads(args.overrides)}
    pairs = load_examples(args.pairs)
    if not pairs:
        raise SystemExit("No preference pairs to train on")
    print("RESULT " + json.dumps(train(args.model, args.sft_adapter, pairs, args.output, args.seed, settings,
                                       args.max_steps)))


if __name__ == "__main__":
    main()
