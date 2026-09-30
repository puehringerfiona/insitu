"""
Run pyOWT classification for every campaign directory.

For each campaign under --campaign-root, this runner selects one final-QC
hyperspectral metadata file from visualization_analysis/quality_control and
writes outputs to:

    <campaign>/visualization_analysis/owt/<filter_mode>

Selection rules:
  1. Use files matching *_metadata_with_Rrs*_final_qc.csv in
     visualization_analysis/quality_control.
  2. Exclude band-convolved files containing "conv" in the filename.
  3. Prefer non-current files over *_current.csv variants.
  4. Prefer Landsat files over Sentinel files when several hyperspectral files
     are otherwise equivalent: L9, then L8, then S2B, then S2A.
  5. Prefer the file with the most Rrs wavelength columns.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyowt_classification import parse_wavelength_from_column


SENSOR_PRIORITY = ["L9", "L8", "S2B", "S2A"]
FILTER_MODES = {
    "all_measurements": None,
    "hard_filter_retained": "hard_filter_retain",
    "hard_filter_strict_retained": "hard_filter_retain_strict",
}


@dataclass(frozen=True)
class CampaignInput:
    campaign_dir: Path
    input_csv: Path
    output_dir: Path
    spectral_column_count: int
    filter_mode: str
    quality_column: str | None


def count_rrs_columns(path: Path, rrs_column_regex: str, wl_min: float, wl_max: float) -> int:
    header = pd.read_csv(path, sep=None, engine="python", nrows=0).columns
    count = 0
    for column in header:
        wl = parse_wavelength_from_column(column, rrs_column_regex)
        if wl is not None and wl_min <= wl <= wl_max:
            count += 1
    return count


def sensor_rank(path: Path) -> int:
    name = path.stem.upper()
    for idx, sensor in enumerate(SENSOR_PRIORITY):
        if re.search(rf"(?:^|_)({sensor})(?:_|$)", name):
            return idx
    return len(SENSOR_PRIORITY)


def file_score(path: Path, spectral_count: int) -> tuple[int, int, int, str]:
    is_current = "current" in path.stem.lower()
    return (
        spectral_count,
        0 if not is_current else -1,
        -sensor_rank(path),
        path.name,
    )


def discover_campaign_inputs(
    campaign_root: Path,
    rrs_column_regex: str,
    wl_min: float,
    wl_max: float,
) -> list[CampaignInput]:
    campaign_inputs: list[CampaignInput] = []

    for campaign_dir in sorted(campaign_root.iterdir()):
        if not campaign_dir.is_dir() or not re.match(r"\d{8}_[A-Za-z0-9]+$", campaign_dir.name):
            continue

        quality_control = campaign_dir / "visualization_analysis" / "quality_control"
        if not quality_control.exists():
            continue

        candidates: list[tuple[Path, int]] = []
        for path in sorted(quality_control.glob("*metadata_with_Rrs*_final_qc.csv")):
            if "conv" in path.name.lower():
                continue
            spectral_count = count_rrs_columns(path, rrs_column_regex, wl_min, wl_max)
            if spectral_count >= 5:
                candidates.append((path, spectral_count))

        if not candidates:
            print(f"SKIP {campaign_dir.name}: no non-convolved hyperspectral metadata file found")
            continue

        selected, spectral_count = max(candidates, key=lambda item: file_score(item[0], item[1]))
        for filter_mode, quality_column in FILTER_MODES.items():
            campaign_inputs.append(
                CampaignInput(
                    campaign_dir=campaign_dir,
                    input_csv=selected,
                    output_dir=campaign_dir / "visualization_analysis" / "owt" / filter_mode,
                    spectral_column_count=spectral_count,
                    filter_mode=filter_mode,
                    quality_column=quality_column,
                )
            )

    return campaign_inputs


def build_classifier_command(args: argparse.Namespace, item: CampaignInput) -> list[str]:
    command = [
        sys.executable,
        str(args.classifier_script),
        "--input-csv",
        str(item.input_csv),
        "--output-dir",
        str(item.output_dir),
        "--wl-min",
        str(args.wl_min),
        "--wl-max",
        str(args.wl_max),
        "--rrs-column-regex",
        args.rrs_column_regex,
        "--owt-version",
        args.owt_version,
        "--membership-threshold",
        str(args.membership_threshold),
    ]

    if args.sensor is not None:
        command.extend(["--sensor", args.sensor])
    quality_column = args.quality_column if args.quality_column is not None else item.quality_column
    if quality_column is not None:
        command.extend(["--quality-column", quality_column])
    if args.quality_min is not None:
        command.extend(["--quality-min", str(args.quality_min)])
    if args.reject_negative_rrs:
        command.append("--reject-negative-rrs")
    if args.no_plots:
        command.append("--no-plots")

    return command


def build_arg_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run pyOWT classification for every campaign.")
    parser.add_argument("--campaign-root", type=Path, default=Path(r"C:\Users\puehrifi\Documents\insitu"))
    parser.add_argument("--classifier-script", type=Path, default=here / "pyowt_classification.py")
    parser.add_argument("--dry-run", action="store_true", help="Print selected inputs and output directories without running classification.")
    parser.add_argument("--wl-min", type=float, default=400)
    parser.add_argument("--wl-max", type=float, default=800)
    parser.add_argument("--rrs-column-regex", default=r"^(?:Rrs_)?\d{3}(?:\.\d+)?(?:nm)?$")
    parser.add_argument("--quality-column", default=None)
    parser.add_argument("--quality-min", type=float, default=None)
    parser.add_argument(
        "--filter-modes",
        nargs="+",
        choices=sorted(FILTER_MODES),
        default=sorted(FILTER_MODES),
        help="Named measurement cohorts to classify.",
    )
    parser.add_argument("--sensor", default=None, help="Optional pyOWT sensor name or alias for multispectral inputs.")
    parser.add_argument("--owt-version", choices=["v01", "v02"], default="v01")
    parser.add_argument("--membership-threshold", type=float, default=0.0001)
    parser.add_argument("--reject-negative-rrs", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    items = [
        item
        for item in discover_campaign_inputs(args.campaign_root, args.rrs_column_regex, args.wl_min, args.wl_max)
        if item.filter_mode in set(args.filter_modes)
    ]

    if not items:
        raise SystemExit("No campaign inputs found.")

    print("Selected campaign inputs:")
    for item in items:
        filter_text = item.quality_column or "none"
        print(
            f"  {item.campaign_dir.name} [{item.filter_mode}, filter={filter_text}]: "
            f"{item.input_csv.name} -> {item.output_dir} ({item.spectral_column_count} bands)"
        )

    if args.dry_run:
        return

    failures: list[tuple[str, int]] = []
    for item in items:
        print(f"\nRUN {item.campaign_dir.name} [{item.filter_mode}]")
        item.output_dir.mkdir(parents=True, exist_ok=True)
        command = build_classifier_command(args, item)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append((item.campaign_dir.name, completed.returncode))

    if failures:
        print("\nFailures:")
        for campaign, returncode in failures:
            print(f"  {campaign}: return code {returncode}")
        raise SystemExit(1)

    print("\nAll campaign OWT classifications completed.")


if __name__ == "__main__":
    main()
