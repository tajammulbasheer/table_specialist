"""Measure the repo's Spark-Scala sandbox: seconds per program, and throughput with parallel workers.

The Blueprint assumed 0.8 s per Pandas run. A fresh JVM per program, as the repo requires, will be far slower, so
the budget for Spark-Scala validation has to come from this measurement.
"""
import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ts.repo import load_sandbox  # noqa: E402

sandbox = load_sandbox()
ensure_sandbox, run_generated_code = sandbox.ensure_sandbox, sandbox.run_generated_code

TABLE = "Region,Year,Sales\n" + "".join(f"r{i % 5},{2000 + i % 20},{i * 3}\n" for i in range(100))
PROGRAM = """val df = spark.read.option("header", "true").csv("data.csv")
val finalResult = df.groupBy("Region").agg(sum("Sales").alias("Total"))
finalResult.coalesce(1).write.option("header", "true").csv("result")
"""
EXPECTED_ROWS = 5


def run_one(_):
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        (work / "data.csv").write_text(TABLE)
        (work / "query.scala").write_text(PROGRAM)
        start = time.perf_counter()
        result = run_generated_code(str(work), "scala", "query.scala", num_cpu=1)
        seconds = time.perf_counter() - start
        parts = list((work / "result_scala").glob("*.csv")) if (work / "result_scala").exists() else []
        rows = len(parts[0].read_text().splitlines()) - 1 if len(parts) == 1 else -1
        return {"seconds": seconds, "ok": result.returncode == 0 and rows == EXPECTED_ROWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ensure_sandbox("scala")
    single = run_one(0)
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        batch = list(pool.map(run_one, range(args.runs)))
    wall = time.perf_counter() - start

    runs_per_hour = args.runs / wall * 3600
    result = {"single_run_seconds": round(single["seconds"], 1), "single_run_ok": single["ok"],
              "workers": args.workers, "runs": args.runs, "ok_runs": sum(r["ok"] for r in batch),
              "mean_seconds_per_run": round(sum(r["seconds"] for r in batch) / args.runs, 1),
              "wall_seconds": round(wall, 1), "runs_per_hour": round(runs_per_hour),
              "hours_for_30000_runs": round(30000 / runs_per_hour, 1)}
    print("RESULT " + json.dumps(result))
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
