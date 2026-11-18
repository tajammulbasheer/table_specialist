"""Build the training-table corpus from GitTables, with the limits in configs/base.yaml.

Writes data/corpus/tables/<id>.csv, results/ids/corpus.txt and results/corpus_manifest.json.

Steps:
  1. Verify every zip against Zenodo's MD5, and refuse to build from an incomplete download.
  2. Walk all tables in a seeded shuffle (the repo's generator also shuffles with seed 42).
  3. Keep tables within the size limits, using the repo's table format to count tokens.
  4. Drop tables whose headers nearly match a benchmark table (contamination), and keep at most
     max_per_header_group tables with nearly the same headers, since GitTables has many CSVs sharing one schema.
"""
import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import tiktoken
import yaml

from ts.repo import BENCH_ROOT, REPO, ROOT

TOKENIZER_MODEL = "gpt-3.5-turbo"  # the repo counts prompt tokens with this tokenizer
MAX_EMPTY_CELL_SHARE = 0.5


def repo_serializer():
    finetune = str(REPO / "Finetune")
    if finetune not in sys.path:
        sys.path.insert(0, finetune)
    from data_generator.base.table_serializer import MarkdownShortLineSerializer
    return MarkdownShortLineSerializer()


def normalise_header(name):
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def header_set(columns):
    return frozenset(normalise_header(c) for c in columns)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


class HeaderIndex:
    """Find near-duplicate header sets by comparing only sets that share a column name."""

    def __init__(self, threshold):
        self.threshold, self.sets, self.by_name = threshold, [], defaultdict(list)

    def add(self, headers):
        position = len(self.sets)
        self.sets.append(headers)
        for name in headers:
            self.by_name[name].append(position)

    def count(self, headers):
        candidates = Counter(p for name in headers for p in self.by_name.get(name, ()))
        # A set must share at least this many names to reach the threshold.
        needed = self.threshold * len(headers)
        return sum(1 for p, shared in candidates.items()
                   if shared >= needed and jaccard(headers, self.sets[p]) >= self.threshold)

    def matches(self, headers):
        return self.count(headers) > 0


def verify_zips(raw):
    record = json.loads((raw / "record.json").read_text())
    problems, sizes = [], 0
    for file in record["files"]:
        path = raw / file["key"]
        if not path.exists() or path.stat().st_size != file["size"]:
            problems.append(f"{file['key']}: incomplete")
            continue
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if f"md5:{digest.hexdigest()}" != file["checksum"]:
            problems.append(f"{file['key']}: checksum differs")
        sizes += file["size"]
    return {"files": len(record["files"]), "bytes": sizes, "doi": record.get("doi"),
            "version": record["metadata"].get("version"), "license": record["metadata"].get("license"),
            "problems": problems}


def benchmark_headers(threshold, bench_root=BENCH_ROOT):
    """Collect the header set of every benchmark table: WikiSQL, WikiTQ, Spider and BIRD."""
    index, count = HeaderIndex(threshold), 0
    patterns = ["WikiSQL_tables/*.csv", "WikiQA/*/data.csv", "spider/*/data.csv", "bird/*/data.csv"]
    seen = set()
    for pattern in patterns:
        for path in Path(bench_root).glob(pattern):
            headers = header_set(pd.read_csv(path, nrows=0).columns)
            if headers and headers not in seen:
                seen.add(headers)
                index.add(headers)
                count += 1
    return index, count


def rejection_reason(frame, limits, serializer, encoder):
    """Return why a table fails the size or quality limits, or None if it passes."""
    rows, columns = frame.shape
    if not limits["min_columns"] <= columns <= limits["max_columns"]:
        return "columns"
    if not limits["min_rows"] <= rows <= limits["max_rows"]:
        return "rows"
    if any(not str(c).strip() or str(c).startswith("Unnamed") for c in frame.columns):
        return "blank_header"
    # SQLite ignores case in column names, so "Logon" and "logon" cannot both exist in one table.
    if len({normalise_header(c) for c in frame.columns}) != columns:
        return "duplicate_header"
    empty = frame.isna() | frame.astype(str).apply(lambda s: s.str.strip() == "")
    if empty.to_numpy().mean() > MAX_EMPTY_CELL_SHARE:
        return "mostly_empty"
    if len(encoder.encode(serializer.serialize_df(frame), disallowed_special=())) > limits["max_serialised_tokens"]:
        return "too_many_tokens"
    return None


def list_tables(raw):
    tables = []
    for path in sorted(raw.glob("*_licensed.zip")):
        with zipfile.ZipFile(path) as archive:
            tables.extend((path.name, name) for name in sorted(archive.namelist()) if name.endswith(".parquet"))
    return tables


_worker = {}


def _init_worker(raw, limits):
    _worker.update(raw=Path(raw), limits=limits, archives={}, serializer=repo_serializer(),
                   encoder=tiktoken.encoding_for_model(TOKENIZER_MODEL))


def _examine(item):
    """Read one table and apply the size and quality limits. Returns (item, reason, csv text, columns)."""
    zip_name, member = item
    archives = _worker["archives"]
    if zip_name not in archives:  # opening a zip reads its whole index, so do it once per worker
        archives[zip_name] = zipfile.ZipFile(_worker["raw"] / zip_name)
    try:
        # Column and row limits need only the file's footer, so reject on those without decoding the table.
        data = io.BytesIO(archives[zip_name].read(member))
        meta = pq.read_metadata(data)
        limits = _worker["limits"]
        if not limits["min_columns"] <= meta.num_columns <= limits["max_columns"]:
            return item, "columns", None, None
        if not limits["min_rows"] <= meta.num_rows <= limits["max_rows"]:
            return item, "rows", None, None
        data.seek(0)
        frame = pq.read_table(data).to_pandas()
    except Exception:  # GitTables is real-world data, and a few files cannot be read
        return item, "unreadable", None, None
    reason = rejection_reason(frame, limits, _worker["serializer"], _worker["encoder"])
    if reason:
        return item, reason, None, None
    return item, None, frame.to_csv(index=False), list(frame.columns)


def build(raw, out_dir, limits, seed, ids_file, manifest_file, bench_root=BENCH_ROOT, verify=True, workers=24):
    raw = Path(raw)
    verification = verify_zips(raw) if verify else {"skipped": True, "problems": []}
    if verification["problems"]:
        raise RuntimeError(f"GitTables download has problems: {verification['problems'][:5]}")

    tables_dir = Path(out_dir) / "tables"
    shutil.rmtree(tables_dir, ignore_errors=True)
    tables_dir.mkdir(parents=True)

    all_tables = list_tables(raw)
    random.Random(seed).shuffle(all_tables)

    threshold = limits["dedupe_header_jaccard"]
    contamination, benchmark_tables = benchmark_headers(threshold, bench_root)
    kept_headers = HeaderIndex(threshold)

    rejected, ids, topics = Counter(), [], Counter()
    examined = 0
    # Workers do the reading and limit checks. imap keeps the shuffled order, and the header filters run here in that
    # order, so the result does not depend on the number of workers.
    # spawn, not fork: this process is multi-threaded by now, and forking one can deadlock the children.
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(raw, limits)) as pool:
        for (zip_name, member), reason, csv_text, columns in pool.imap(_examine, all_tables, chunksize=64):
            if len(ids) == limits["size"]:
                break
            examined += 1
            if reason:
                rejected[reason] += 1
                continue
            headers = header_set(columns)
            if contamination.matches(headers):
                rejected["matches_benchmark"] += 1
                continue
            if kept_headers.count(headers) >= limits["max_per_header_group"]:
                rejected["duplicate_of_kept"] += 1
                continue
            kept_headers.add(headers)
            table_id = f"gt_{len(ids):05d}"
            (tables_dir / f"{table_id}.csv").write_text(csv_text)
            ids.append(f"{table_id}\t{zip_name}\t{member}")
            topics[zip_name] += 1
        pool.terminate()

    Path(ids_file).parent.mkdir(parents=True, exist_ok=True)
    Path(ids_file).write_text("\n".join(ids) + "\n")
    manifest = {"source": verification, "tables_available": len(all_tables), "tables_examined": examined,
                "tables_kept": len(ids), "target": limits["size"], "seed": seed, "limits": limits,
                "benchmark_header_sets_checked": benchmark_tables, "rejected": dict(rejected.most_common()),
                "topics_used": len(topics), "largest_topic_share": round(max(topics.values(), default=0) / max(len(ids), 1), 4)}
    Path(manifest_file).write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/gittables"))
    ap.add_argument("--out", default=str(ROOT / "data/corpus"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    manifest = build(args.raw, args.out, config["corpus"], args.seed, ROOT / "results/ids/corpus.txt",
                     ROOT / "results/corpus_manifest.json")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
