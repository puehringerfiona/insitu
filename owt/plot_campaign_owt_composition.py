"""
Plot campaign-level pyOWT composition as stacked bars.

The script reads each campaign's
visualization_analysis/owt/<owt-subdir>/owt_classified_spectra.csv and keeps only
the spectra that pass the hard filter (hard_filter_retain) and that fall on the
transects configured in campaign_transects.CAMPAIGN_TRANSECTS. Composition is
counted from that cohort, so the figure matches the final QC selection rather
than the unrestricted classification.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DEFAULT_CAMPAIGN_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
DEFAULT_OUTPUT_DIR = DEFAULT_CAMPAIGN_ROOT / "all" / "visualization_analysis" / "owt"

QC_MODULE_DIR = Path(
    r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\quality_control"
)
if str(QC_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(QC_MODULE_DIR))

from campaign_transects import CAMPAIGN_TRANSECTS

# Classify from the unfiltered cohort and apply both selections here, so the
# hard filter and the transect restriction are always enforced together.
DEFAULT_OWT_SUBDIR = "all_measurements"
CLASSIFIED_CSV = "owt_classified_spectra.csv"
REQUIRED_COLUMNS = ["owt_label", "transect_nr", "hard_filter_retain"]

# Publication figure geometry: fixed 10 cm column width with 8 pt as the hard
# floor for every text element. Helvetica is not installed on every machine, so
# fall back to Arial (metrically identical) and then DejaVu Sans.
FIG_WIDTH_CM, FIG_HEIGHT_CM = 10.0, 9.0
CM_TO_IN = 1 / 2.54
FONT_FAMILY = ["Helvetica", "Arial", "DejaVu Sans"]
BASE_FONTSIZE = 8
AXIS_LABEL_FONTSIZE = 9
LEGEND_NCOL = 3
LEGEND_LABELS = ["3a", "3b", "4a"]

# Bottom margin budget in cm, reserved so the 10 cm figure width is never
# rewritten by a tight bounding box: rotated tick labels, the x axis label, the
# legend title, and one entry per legend row.
TICK_LABEL_CM = 1.95
XLABEL_CM = 0.55
LEGEND_GAP_CM = 0.3
LEGEND_TITLE_CM = 0.5
LEGEND_ROW_CM = 0.42
BOTTOM_PAD_CM = 0.15

OWT_ORDER = ["1", "2", "3a", "3b", "4a", "4b", "5a", "5b", "6", "7", "NaN"]
OWT_COLORS = {
    "1": "#4C6A9C",
    "2": "#D8C76B",
    "3a": "#D89A54",
    "3b": "#6FB6C9",
    "4a": "#7DB76D",
    "4b": "#8C6BB1",
    "5a": "#3E4A89",
    "5b": "#B85C5C",
    "6": "#B8794D",
    "7": "#5A8F8C",
    "NaN": "#9A9A9A",
}


def campaign_sort_key(path: Path) -> tuple[str, str]:
    match = re.match(r"^(\d{8})_([A-Za-z0-9]+)$", path.name)
    if not match:
        return path.name, ""
    return match.group(1), match.group(2)


def campaign_label(campaign: str) -> str:
    """Compact campaign alias, e.g. 20230610_CST -> CST20230610."""
    match = re.match(r"^(\d{8})_([A-Za-z0-9]+)$", campaign)
    if not match:
        return campaign
    return f"{match.group(2)}{match.group(1)}"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": BASE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "axes.titlesize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": BASE_FONTSIZE,
            "ytick.labelsize": BASE_FONTSIZE,
            "legend.fontsize": BASE_FONTSIZE,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def ordered_labels(composition: pd.DataFrame) -> list[str]:
    present = set(composition["owt_label"].astype(str))
    labels_present = [label for label in OWT_ORDER if label in present]
    return labels_present + sorted(present - set(labels_present))


def draw_stacked_bars(ax, pivot: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """Draw the stacked bars and return the per-campaign stack tops."""
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index), dtype=float)
    for label in labels:
        values = pivot[label].to_numpy(dtype=float)
        if not np.any(values):
            continue
        ax.bar(
            x,
            values,
            0.72,
            bottom=bottom,
            label=f"OWT {label}" if label != "NaN" else "Not classifiable",
            color=OWT_COLORS.get(label, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values
    return bottom


def finish_axes(fig, ax, campaigns: list[str], labels: list[str]) -> None:
    """Apply the shared axis, tick and legend styling to a stacked bar axes."""
    ax.set_xlabel("Campaign", labelpad=4)
    ax.set_xticks(np.arange(len(campaigns)))
    ax.set_xticklabels(
        [campaign_label(campaign) for campaign in campaigns],
        rotation=90,
        fontsize=BASE_FONTSIZE,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)

    legend_labels = [label for label in LEGEND_LABELS if label in OWT_COLORS]
    ncol = len(legend_labels)
    legend_rows = 1 if legend_labels else 0

    # The legend is anchored below the rotated tick labels and the x axis label;
    # the reserved band must cover that offset plus the legend itself, otherwise
    # the last legend row falls off the canvas.
    legend_gap_cm = TICK_LABEL_CM + XLABEL_CM + LEGEND_GAP_CM
    bottom_cm = (
        legend_gap_cm + LEGEND_TITLE_CM + legend_rows * LEGEND_ROW_CM + BOTTOM_PAD_CM
    )
    bottom = bottom_cm / FIG_HEIGHT_CM
    # Reserve that band below the axes instead of letting a tight bbox grow the
    # canvas, so the saved figure stays exactly FIG_WIDTH_CM wide.
    fig.subplots_adjust(left=0.125, right=0.985, top=0.985, bottom=bottom)

    axes_height_cm = FIG_HEIGHT_CM * (0.985 - bottom)
    legend_handles = [
        Patch(
            facecolor=OWT_COLORS[label],
            edgecolor="white",
            linewidth=0.5,
            label=f"OWT {label}",
        )
        for label in legend_labels
    ]
    ax.legend(
        handles=legend_handles,
        title="OWT classes",
        loc="upper center",
        bbox_to_anchor=(0.5, -(legend_gap_cm / axes_height_cm)),
        ncol=ncol,
        frameon=False,
        fontsize=BASE_FONTSIZE,
        title_fontsize=BASE_FONTSIZE,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.35,
        borderaxespad=0.0,
    )


def find_classified_files(campaign_root: Path, owt_subdir: str) -> list[Path]:
    """Locate the per-spectrum classification CSV for every configured campaign."""
    files: list[Path] = []
    for campaign_dir in sorted(campaign_root.iterdir(), key=campaign_sort_key):
        if not campaign_dir.is_dir() or campaign_dir.name not in CAMPAIGN_TRANSECTS:
            continue
        owt_dir = campaign_dir / "visualization_analysis" / "owt"
        if owt_subdir not in {"", "."}:
            owt_dir = owt_dir / owt_subdir
        classified = owt_dir / CLASSIFIED_CSV
        if classified.exists():
            files.append(classified)
    return files


def campaign_from_count_path(path: Path) -> str:
    for parent in path.parents:
        if re.match(r"^\d{8}_[A-Za-z0-9]+$", parent.name):
            return parent.name
    raise ValueError(f"Could not infer campaign directory from {path}")


def normalize_owt_label(value: object) -> str:
    label = str(value).strip()
    return "NaN" if not label or label.lower() in {"nan", "none"} else label


def load_retained_labels(path: Path, campaign: str) -> pd.Series:
    """Return the OWT labels passing the hard filter and the campaign transects."""
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS, keep_default_na=False, low_memory=False)

    retained = df["hard_filter_retain"].astype(str).str.strip().str.lower() == "true"
    transects = CAMPAIGN_TRANSECTS[campaign]
    on_transect = pd.to_numeric(df["transect_nr"], errors="coerce").isin(transects)

    return df.loc[retained & on_transect, "owt_label"].map(normalize_owt_label)


def build_composition(classified_files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in classified_files:
        campaign = campaign_from_count_path(path)
        labels = load_retained_labels(path, campaign)
        if labels.empty:
            print(f"SKIP {campaign}: no spectra pass hard_filter_retain on transects {CAMPAIGN_TRANSECTS[campaign]}")
            continue
        counts = labels.value_counts()
        total = int(counts.sum())
        for label, n in counts.items():
            rows.append(
                {
                    "campaign": campaign,
                    "owt_label": label,
                    "n": int(n),
                    "fraction": int(n) / total,
                }
            )
    if not rows:
        raise ValueError("No OWT count rows found.")
    return pd.DataFrame(rows)


def plot_stacked_counts(composition: pd.DataFrame, output_png: Path) -> None:
    apply_plot_style()
    campaigns = list(dict.fromkeys(composition["campaign"].astype(str)))
    labels = ordered_labels(composition)

    pivot = (
        composition.pivot_table(index="campaign", columns="owt_label", values="n", aggfunc="sum", fill_value=0)
        .reindex(index=campaigns, columns=labels, fill_value=0)
        .astype(float)
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_CM * CM_TO_IN, FIG_HEIGHT_CM * CM_TO_IN))
    totals = draw_stacked_bars(ax, pivot, labels)

    offset = max(totals.max() * 0.015, 0.5) if len(totals) else 0.5
    for xi, total in enumerate(totals):
        ax.text(
            xi, total + offset, f"n={int(total)}",
            ha="center", va="bottom", rotation=90, fontsize=BASE_FONTSIZE,
        )

    ax.set_ylabel("Number of spectra", labelpad=4)
    if len(totals) and totals.max() > 0:
        ax.set_ylim(0, totals.max() * 1.30)
    finish_axes(fig, ax, campaigns, labels)
    fig.savefig(output_png, dpi=600)
    plt.close(fig)


def plot_stacked_percent(composition: pd.DataFrame, output_png: Path) -> None:
    apply_plot_style()
    campaigns = list(dict.fromkeys(composition["campaign"].astype(str)))
    labels = ordered_labels(composition)

    pivot = (
        composition.pivot_table(index="campaign", columns="owt_label", values="fraction", aggfunc="sum", fill_value=0)
        .reindex(index=campaigns, columns=labels, fill_value=0)
        .astype(float)
        * 100
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_CM * CM_TO_IN, FIG_HEIGHT_CM * CM_TO_IN))
    draw_stacked_bars(ax, pivot, labels)

    totals = (
        composition.pivot_table(index="campaign", values="n", aggfunc="sum", fill_value=0)
        .reindex(index=campaigns, fill_value=0)["n"]
        .to_numpy(dtype=float)
    )
    for xi, total in enumerate(totals):
        ax.text(
            xi, 102, f"n={int(total)}",
            ha="center", va="bottom", rotation=90, fontsize=BASE_FONTSIZE,
        )

    ax.set_ylabel("Spectra [%]", labelpad=4)
    ax.set_ylim(0, 125)
    ax.set_yticks(np.arange(0, 101, 20))
    finish_axes(fig, ax, campaigns, labels)
    fig.savefig(output_png, dpi=600)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a stacked bar chart of pyOWT composition by campaign.")
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-percent-png", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--owt-subdir",
        default=DEFAULT_OWT_SUBDIR,
        help=(
            "Subfolder below each campaign's visualization_analysis/owt directory holding "
            f"{CLASSIFIED_CSV}. Default: {DEFAULT_OWT_SUBDIR}, since the hard filter and the "
            "transect restriction are applied by this script."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_png = args.output_png or args.output_dir / "owt_campaign_composition_stacked_counts.png"
    output_percent_png = args.output_percent_png or args.output_dir / "owt_campaign_composition_stacked_percent.png"
    output_csv = args.output_csv or args.output_dir / "owt_campaign_composition_counts.csv"

    classified_files = find_classified_files(args.campaign_root, args.owt_subdir)
    if not classified_files:
        raise SystemExit(f"No {CLASSIFIED_CSV} files found below {args.campaign_root}")

    composition = build_composition(classified_files)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    composition.to_csv(output_csv, index=False)
    plot_stacked_counts(composition, output_png)
    plot_stacked_percent(composition, output_percent_png)

    print(f"Read {len(classified_files)} campaign classification files.")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_percent_png}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
