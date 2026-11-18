"""Corpus limits, contamination and duplicate filters, and the refusal to build from a bad download."""
import hashlib
import io
import json
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tiktoken

from ts.corpus.build import (HeaderIndex, build, header_set, jaccard, rejection_reason, repo_serializer, verify_zips)

LIMITS = {"size": 3, "min_columns": 5, "max_columns": 30, "min_rows": 10, "max_rows": 200,
          "max_serialised_tokens": 2000, "dedupe_header_jaccard": 0.9, "max_per_header_group": 1}


def table(columns=5, rows=20, prefix="c"):
    return pd.DataFrame({f"{prefix}{i}": [f"v{r}" for r in range(rows)] for i in range(columns)})


def reason(frame, limits=LIMITS):
    return rejection_reason(frame, limits, repo_serializer(), tiktoken.encoding_for_model("gpt-3.5-turbo"))


def test_good_table_passes():
    assert reason(table()) is None


@pytest.mark.parametrize("frame,expected", [
    (table(columns=4), "columns"), (table(columns=31), "columns"),
    (table(rows=9), "rows"), (table(rows=201), "rows"),
])
def test_size_limits(frame, expected):
    assert reason(frame) == expected


def test_limits_are_inclusive():
    assert reason(table(columns=5, rows=10)) is None
    assert reason(table(columns=30, rows=10)) is None


def test_quality_limits():
    assert reason(table().rename(columns={"c0": "Unnamed: 0"})) == "blank_header"
    assert reason(table().rename(columns={"c1": "c0"})) == "duplicate_header"
    assert reason(pd.DataFrame({f"c{i}": [""] * 20 for i in range(5)})) == "mostly_empty"
    long_cells = pd.DataFrame({f"c{i}": ["word " * 200] * 20 for i in range(5)})
    assert reason(long_cells) == "too_many_tokens"


def test_headers_differing_only_by_case_are_duplicates():
    frame = table().rename(columns={"c0": "Logon", "c1": "logon"})
    assert reason(frame) == "duplicate_header"


def test_header_matching_ignores_case_and_spacing():
    assert header_set(["Team  Name", "YEAR"]) == header_set(["team name", "year"])


def test_header_index_threshold():
    index = HeaderIndex(0.9)
    names = [f"c{i}" for i in range(20)]
    index.add(frozenset(names))
    assert index.matches(frozenset(names))
    assert index.matches(frozenset(names + ["extra"]))         # Jaccard 20/21 = 0.95
    assert not index.matches(frozenset(names[:10] + ["x", "y"]))
    assert not index.matches(frozenset(["a", "b"]))
    assert jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)


def corpus_zip(raw, name, tables):
    path = raw / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, frame in tables.items():
            buffer = io.BytesIO()
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), buffer)
            archive.writestr(member, buffer.getvalue())
    return {"key": name, "size": path.stat().st_size,
            "checksum": "md5:" + hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()}


@pytest.fixture()
def world(tmp_path):
    raw, bench = tmp_path / "raw", tmp_path / "bench"
    raw.mkdir()
    (bench / "spider" / "case1").mkdir(parents=True)
    table(prefix="bench").to_csv(bench / "spider" / "case1" / "data.csv", index=False)
    entry = corpus_zip(raw, "topic_tables_licensed.zip", {
        "a.parquet": table(prefix="a"),
        "b.parquet": table(prefix="b"),
        "duplicate_of_a.parquet": table(prefix="a"),
        "contaminated.parquet": table(prefix="bench"),
        "small.parquet": table(columns=2, prefix="s"),
    })
    (raw / "record.json").write_text(json.dumps({"files": [entry], "doi": "x", "metadata": {"version": "1"}}))
    return raw, bench, tmp_path


def run(world, **limits):
    raw, bench, tmp = world
    return build(raw, tmp / "corpus", {**LIMITS, **limits}, 42, tmp / "ids.txt", tmp / "manifest.json", bench_root=bench)


def test_build_filters_everything_it_should(world):
    manifest = run(world, size=100)
    assert manifest["tables_kept"] == 2
    assert manifest["rejected"] == {"columns": 1, "matches_benchmark": 1, "duplicate_of_kept": 1}
    assert manifest["source"]["problems"] == []
    kept = (world[2] / "ids.txt").read_text().splitlines()
    assert len(kept) == 2 and all(len(line.split("\t")) == 3 for line in kept)
    assert len(list((world[2] / "corpus" / "tables").glob("*.csv"))) == 2


def test_header_group_limit_allows_that_many_near_duplicates(world):
    manifest = run(world, size=100, max_per_header_group=2)
    assert manifest["tables_kept"] == 3  # a, its duplicate, and b
    assert "duplicate_of_kept" not in manifest["rejected"]
    assert manifest["rejected"]["matches_benchmark"] == 1  # the benchmark filter stays strict


def test_build_stops_at_the_target_size(world):
    assert run(world, size=1)["tables_kept"] == 1


def test_build_is_reproducible(world):
    run(world, size=100)
    first = (world[2] / "ids.txt").read_text()
    run(world, size=100)
    assert (world[2] / "ids.txt").read_text() == first


def test_result_does_not_depend_on_worker_count(world):
    raw, bench, tmp = world
    outputs = []
    for workers in (1, 4):
        build(raw, tmp / "corpus", {**LIMITS, "size": 100}, 42, tmp / "ids.txt", tmp / "manifest.json",
              bench_root=bench, workers=workers)
        outputs.append((tmp / "ids.txt").read_text())
    assert outputs[0] == outputs[1]


def test_build_refuses_a_bad_download(world):
    raw = world[0]
    path = raw / "topic_tables_licensed.zip"
    data = bytearray(path.read_bytes())
    data[10] ^= 0xFF
    path.write_bytes(data)
    assert "checksum differs" in verify_zips(raw)["problems"][0]
    with pytest.raises(RuntimeError):
        run(world)
    path.write_bytes(data[:-5])
    assert "incomplete" in verify_zips(raw)["problems"][0]
