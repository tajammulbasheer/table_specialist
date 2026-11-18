"""Build the dev sets used only to tune extension 2's DPO settings. Never used for reported scores.

Each comes from a split the test sets do not use: WikiSQL dev (test uses test), Spider dev (test uses test), WikiTQ
random-split-1-dev (test uses pristine-unseen-tables, whose tables appear in no other split). BIRD has no unused split
downloaded: its test set is BIRD's dev split. Writes data/bench/NL2SQL/benchmark/dev/ and results/ids/dev/.

  python -m ts.bench.dev
"""
import json

from ts.bench import spider, wikisql, wikitq
from ts.repo import ROOT

DEV_ROOT = ROOT / "data" / "bench" / "NL2SQL" / "benchmark" / "dev"
DEV_IDS = ROOT / "results" / "ids" / "dev"
DEV_SIZE = 300  # ours: per set, for WikiSQL and WikiTQ; Spider keeps every single-table dev question
SEED = 42       # configs/base.yaml benchmarks.sampling_seed


def main():
    DEV_IDS.mkdir(parents=True, exist_ok=True)
    manifests = {
        "WikiSQL": wikisql.build(ROOT / "data/raw/wikisql/data.tar.bz2", DEV_SIZE, SEED, DEV_ROOT,
                                 DEV_IDS / "wikisql.txt", split="dev"),
        "spider": spider.build(ROOT / "data/raw/spider/spider.zip", DEV_ROOT, DEV_IDS / "spider.txt", split="dev"),
        "WikiQA": wikitq.build(ROOT / "data/raw/wikitq", DEV_SIZE, SEED, DEV_ROOT, DEV_IDS / "wikiqa.txt",
                               split="random-split-1-dev"),
    }
    print(json.dumps({name: {k: v for k, v in m.items() if not isinstance(v, list)} for name, m in manifests.items()},
                     indent=1))


if __name__ == "__main__":
    main()
