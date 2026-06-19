from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_results import _analyze_input_file


SCRIPT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Quantinuum Helios Shor sweep results with the IBM-series metrics."
    )
    parser.add_argument("--input", required=True, help="JSONL results file from run_helios_sweep.py")
    parser.add_argument(
        "--output-csv",
        default=str(SCRIPT_ROOT / "data" / "summary" / "results_summary_helios.csv"),
    )
    parser.add_argument("--figures-dir", default=str(SCRIPT_ROOT / "figures"))
    args = parser.parse_args()

    _analyze_input_file(
        input_path=Path(args.input),
        output_csv=Path(args.output_csv),
        figures_dir=Path(args.figures_dir),
        run_label="quantinuum-helios",
        required=True,
    )


if __name__ == "__main__":
    main()
