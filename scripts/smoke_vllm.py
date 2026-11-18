"""Measure vLLM prefill and decode throughput with prompts shaped like the study's (about 3,000 tokens)."""
import argparse
import json
import random
import time

from vllm import LLM, SamplingParams


def make_prompt_ids(tokenizer, target_len, rng):
    # Table-like text with unique content, so prefix caching can't flatter the numbers.
    words = ["revenue", "region", "2019", "north", "total", "name", "id", "3.14", "active", "city", "|", "\n"]
    text = " ".join(rng.choice(words) + str(rng.randint(0, 9999)) for _ in range(target_len))
    return tokenizer(text, add_special_tokens=False).input_ids[:target_len]


def timed_generate(llm, prompts, params):
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    return time.perf_counter() - start, outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--prompt-tokens", type=int, default=3000)
    ap.add_argument("--output-tokens", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load_start = time.perf_counter()
    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=0.92,
              max_model_len=4096, enable_prefix_caching=True, seed=0)
    load_s = time.perf_counter() - load_start
    tokenizer = llm.get_tokenizer()
    rng = random.Random(0)

    def batch():
        return [{"prompt_token_ids": make_prompt_ids(tokenizer, args.prompt_tokens, rng)}
                for _ in range(args.prompts)]

    prefill_batch = batch()
    prefill_tokens = sum(len(p["prompt_token_ids"]) for p in prefill_batch)
    prefill_s, _ = timed_generate(llm, prefill_batch, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
    prefill_rate = prefill_tokens / prefill_s

    mixed_batch = batch()
    mixed_prompt_tokens = sum(len(p["prompt_token_ids"]) for p in mixed_batch)
    mixed_s, outputs = timed_generate(
        llm, mixed_batch, SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True))
    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    decode_s = max(mixed_s - mixed_prompt_tokens / prefill_rate, 1e-6)

    result = {
        "model": args.model, "dtype": args.dtype, "load_seconds": round(load_s, 1),
        "prefill_tokens_per_s": round(prefill_rate),
        "mixed_seconds": round(mixed_s, 1),
        "decode_tokens_per_s": round(generated / decode_s),
        "doc_assumption_expected": {"prefill_tokens_per_s": 3500, "decode_tokens_per_s": 800},
    }
    print("RESULT " + json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
