"""Fetch a finished Helios execute job by id and write a raw JSONL row.

Reuses the counts-extraction helpers from quantinuum_runtime so the output row
matches the schema produced by run_helios_sweep.py / submit_helios_qir_job.py.
The strict Shor metrics are left to analyze_helios_results.py, which recomputes
them from the stored counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import qnexus as qnx

from quantinuum_runtime import _counts_from_backend_result, _login, _ref_identifier


def _status_field(status: Any, name: str) -> Any:
    value = getattr(status, name, None)
    if value is None and isinstance(status, dict):
        value = status.get(name)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a finished Helios job result by id.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--base", type=int, required=True)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True,
                        help="Existing raw JSONL row to copy static metadata from.")
    parser.add_argument("--submission", type=Path,
                        help="Submission JSON to copy depth/2q/qir refs/max-cost from.")
    parser.add_argument("--login", default="none", choices=["none", "browser", "credentials"])
    args = parser.parse_args()

    _login(qnx, args.login)

    job = qnx.jobs.get(id=args.job_id)
    status = qnx.jobs.status(job)
    print("Job status:", _status_field(status, "status"), "cost:", _status_field(status, "cost"))

    result_ref = qnx.jobs.results(job)[0]
    backend_result = result_ref.download_result()

    template = json.loads(Path(args.template).read_text().splitlines()[0])
    t = template["t"]
    # The Qiskit circuit only measures the t-qubit counting register into the
    # "meas" classical register, so the QIR INT is the t-bit counting value even
    # though required_num_results reports the full circuit width.
    counts = _counts_from_backend_result(backend_result, args.shots, width=t)
    total = sum(counts.values())
    print(f"Recovered {len(counts)} unique bitstrings, {total} shots total")

    row = dict(template)
    row["a"] = args.base
    row["shots"] = args.shots
    row["counts"] = counts
    row["unique_frac"] = (len(counts) / total) if total else 0.0
    row["job_id"] = args.job_id
    row["quantinuum_execute_job_id"] = args.job_id
    raw_status = str(_status_field(status, "status") or "completed")
    # The analyzer gates on a lowercase "completed" token; normalize the enum repr.
    row["run_status"] = "completed" if "completed" in raw_status.lower() else raw_status
    # Derive the real execution runtime from the job timestamps rather than
    # inheriting the template's (which belongs to a different job).
    running = _status_field(status, "running_time")
    completed = _status_field(status, "completed_time")
    runtime_sec = None
    if running is not None and completed is not None:
        runtime_sec = (completed - running).total_seconds()
    row["runtime_sec"] = runtime_sec
    row["quantinuum_execute_runtime_sec"] = runtime_sec

    actual_cost = _status_field(status, "cost")
    result_cost = getattr(result_ref, "cost", None)
    if actual_cost is not None:
        row["quantinuum_actual_cost_hqc"] = float(actual_cost)
    if result_cost is not None:
        row["quantinuum_result_cost_hqc"] = float(result_cost)

    if args.submission and args.submission.exists():
        sub = json.loads(args.submission.read_text())
        for key_src, key_dst in [
            ("depth", "depth"),
            ("two_qubit_gates", "two_qubit_gates"),
            ("qiskit_decomposed_depth", "quantinuum_qiskit_decomposed_depth"),
            ("qiskit_decomposed_instruction_count", "quantinuum_qiskit_decomposed_instruction_count"),
            ("qir_ref", "quantinuum_qir_ref"),
            ("max_cost_hqc", "quantinuum_effective_max_cost_hqc"),
        ]:
            if key_src in sub and sub[key_src] is not None:
                row[key_dst] = sub[key_src]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        fh.write(json.dumps(row) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
