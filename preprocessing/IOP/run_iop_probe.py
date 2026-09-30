from __future__ import annotations

import argparse
from pathlib import Path

from iop_probe import run_lab, run_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process IOP optical probe measurements.")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="Process a vertical profile cast with ACS, VSF, and CTD.")
    profile.add_argument("input_dir", type=Path, help="Extracted measurement folder.")
    profile.add_argument("--run", required=True, help="Run number, for example 292.")
    profile.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    profile.add_argument("--depth-offset-m", type=float, default=0.0, help="Depth offset added to CTD pressure-derived depth.")

    lab = sub.add_parser("lab", help="Process clean-water lab ACS measurements.")
    lab.add_argument("input_dir", type=Path, help="Extracted measurement folder.")
    lab.add_argument("--run", required=True, help="Run number, for example 139.")
    lab.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    lab.add_argument("--trim-percent", type=float, default=30.0, help="Percent removed from each end before statistics.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "profile":
        result = run_profile(args.input_dir, args.run, args.output_dir, args.depth_offset_m)
    elif args.command == "lab":
        result = run_lab(args.input_dir, args.run, args.output_dir, args.trim_percent)
    else:
        parser.error(f"Unknown command: {args.command}")

    print(f"Output directory: {result['output_dir']}")
    print(f"Dashboard: {result['dashboard']}")
    print(f"CSV files: {len(result['csv_files'])}")
    print(f"Plot files: {len(result['plot_files'])}")
    for error in result["errors"]:
        print(f"WARNING: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
