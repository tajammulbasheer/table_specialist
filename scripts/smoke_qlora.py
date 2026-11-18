"""Run a short QLoRA job with the study's settings and measure training speed and peak VRAM."""
import argparse
import json
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer


def make_examples(tokenizer, n, prompt_tokens, completion_tokens):
    rng = random.Random(0)
    words = ["revenue", "region", "2019", "north", "total", "name", "id", "3.14", "active", "city", "|", "\n"]

    def text(length):
        ids = tokenizer(" ".join(rng.choice(words) + str(rng.randint(0, 9999)) for _ in range(length)),
                        add_special_tokens=False).input_ids[:length]
        return tokenizer.decode(ids)

    if tokenizer.chat_template is None:  # base Llama has no chat template, so its examples are plain text
        return Dataset.from_list([{"prompt": text(prompt_tokens), "completion": text(completion_tokens)}
                                  for _ in range(n)])
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": text(prompt_tokens)}],
         "completion": [{"role": "assistant", "content": text(completion_tokens)}]}
        for _ in range(n)
    ])


class StepTimer(TrainerCallback):
    """Skip the first steps, which include model warm-up, when timing."""

    def __init__(self, skip=5):
        self.skip, self.start, self.end, self.steps = skip, None, None, 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.skip:
            self.start = time.perf_counter()
        elif state.global_step > self.skip:
            self.end, self.steps = time.perf_counter(), state.global_step - self.skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--prompt-tokens", type=int, default=1400)
    ap.add_argument("--completion-tokens", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-lora", action="store_true", help="Train on unquantized BF16 weights, as the paper's LoRA does")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = None if args.full_lora else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=quantization)
    dataset = make_examples(tokenizer, args.steps + 10, args.prompt_tokens, args.completion_tokens)
    timer = StepTimer()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
        callbacks=[timer],
        args=SFTConfig(
            output_dir="/tmp/smoke_qlora", max_steps=args.steps, per_device_train_batch_size=1,
            gradient_checkpointing=True, learning_rate=2e-4, lr_scheduler_type="cosine", bf16=True,
            logging_steps=10, save_strategy="no", report_to="none", max_length=4096, seed=0),
    )
    trainer.train()

    tokens_per_example = args.prompt_tokens + args.completion_tokens
    result = {
        "model": args.model,
        "method": "lora" if args.full_lora else "qlora",
        "prompt_tokens": args.prompt_tokens,
        "steps_timed": timer.steps,
        "seconds_per_step": round((timer.end - timer.start) / timer.steps, 3),
        "tokens_per_s_approx": round(tokens_per_example * timer.steps / (timer.end - timer.start)),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "doc_assumption_expected": {"tokens_per_s": 1500, "total_vram_gb": "10-14"},
    }
    print("RESULT " + json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
