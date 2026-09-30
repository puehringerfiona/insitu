"""
Sample campaign-specific depth rasters at transect measurement coordinates.

Input CSVs are expected to contain WGS84 longitude/latitude columns named
``x-coordinate`` and ``y-coordinate``. The script selects the matching depth
raster from ``lake_depth_config.csv``, transforms coordinates to the raster CRS,
and writes/updates a depth column.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal, osr


DEFAULT_INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
DEFAULT_QGIS_ROOT = DEFAULT_INSITU_ROOT / "all" / "qgis_transect_bb"
DEFAULT_DEPTH_ROOT = DEFAULT_QGIS_ROOT / "depth"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("lake_depth_config.csv")
DEFAULT_SOURCE_CRS = "EPSG:4326"
DEFAULT_DEPTH_COLUMN = "depth_m"


@dataclass(frozen=True)
class DepthConfig:
    lake: str
    campaign_date: str
    campaign_code: str
    output_name: str

    @property
    def campaign_id(self) -> str:
        return f"{self.campaign_date}_{self.campaign_code}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add campaign-specific raster depth values to transect CSV files."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Single transect CSV to process. If omitted, use --batch.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output CSV for single-file mode. Default: input file is overwritten.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all configured campaign CSVs found under --insitu-root.",
    )
    parser.add_argument(
        "--insitu-root",
        type=Path,
        default=DEFAULT_INSITU_ROOT,
        help=f"Root containing campaign folders. Default: {DEFAULT_INSITU_ROOT}",
    )
    parser.add_argument(
        "--depth-root",
        type=Path,
        default=DEFAULT_DEPTH_ROOT,
        help=f"Folder containing campaign-specific depth rasters. Default: {DEFAULT_DEPTH_ROOT}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Depth config CSV. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--campaign-date",
        help="Campaign date YYYYMMDD. Required only if it cannot be inferred from --input-file.",
    )
    parser.add_argument(
        "--campaign-code",
        help="Campaign code such as BIE, CST, WAL, or ZRH. Required only if it cannot be inferred.",
    )
    parser.add_argument(
        "--x-column",
        default="x-coordinate",
        help="Longitude column name. Default: x-coordinate.",
    )
    parser.add_argument(
        "--y-column",
        default="y-coordinate",
        help="Latitude column name. Default: y-coordinate.",
    )
    parser.add_argument(
        "--source-crs",
        default=DEFAULT_SOURCE_CRS,
        help="Coordinate CRS of x/y columns. Default: EPSG:4326.",
    )
    parser.add_argument(
        "--depth-column",
        default=DEFAULT_DEPTH_COLUMN,
        help=f"Output depth column name. Default: {DEFAULT_DEPTH_COLUMN}.",
    )
    return parser.parse_args()


def read_depth_config(config_path: Path) -> list[DepthConfig]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config CSV does not exist: {config_path}")

    configs = []
    with config_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"lake", "campaign_date", "campaign_code", "output_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing config columns: {', '.join(sorted(missing))}")

        for row in reader:
            configs.append(
                DepthConfig(
                    lake=row["lake"].strip().lower(),
                    campaign_date=row["campaign_date"].strip(),
                    campaign_code=row["campaign_code"].strip().upper(),
                    output_name=row["output_name"].strip(),
                )
            )

    if not configs:
        raise ValueError(f"No depth config rows found in {config_path}")
    return configs


def infer_campaign(input_file: Path) -> tuple[str | None, str | None]:
    text = str(input_file)
    match = re.search(r"(20\d{6})[_-]([A-Za-z]{3})", text)
    if not match:
        return None, None
    return match.group(1), match.group(2).upper()


def select_config(
    configs: list[DepthConfig],
    input_file: Path,
    campaign_date: str | None,
    campaign_code: str | None,
) -> DepthConfig:
    inferred_date, inferred_code = infer_campaign(input_file)
    campaign_date = campaign_date or inferred_date
    campaign_code = (campaign_code or inferred_code or "").upper()

    if not campaign_date or not campaign_code:
        raise ValueError(
            "Could not infer campaign from input path. Pass --campaign-date and --campaign-code."
        )

    matches = [
        config
        for config in configs
        if config.campaign_date == campaign_date and config.campaign_code == campaign_code
    ]
    if not matches:
        raise ValueError(f"No depth config found for {campaign_date}_{campaign_code}")
    if len(matches) > 1:
        raise ValueError(f"Multiple depth configs found for {campaign_date}_{campaign_code}")
    return matches[0]


def sniff_delimiter(input_file: Path) -> str:
    sample = input_file.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,	,")
        return dialect.delimiter
    except csv.Error:
        return ";"


def create_coordinate_transform(source_crs: str, raster_dataset: gdal.Dataset) -> osr.CoordinateTransformation:
    source = osr.SpatialReference()
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    source.SetFromUserInput(source_crs)

    target = osr.SpatialReference()
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target.ImportFromWkt(raster_dataset.GetProjection())

    return osr.CoordinateTransformation(source, target)


def sample_depths(
    df: pd.DataFrame,
    depth_raster: Path,
    x_column: str,
    y_column: str,
    source_crs: str,
) -> pd.Series:
    if x_column not in df.columns or y_column not in df.columns:
        raise KeyError(
            f"Missing coordinate columns. Need {x_column!r} and {y_column!r}; "
            f"available columns: {df.columns.tolist()}"
        )

    dataset = gdal.Open(str(depth_raster), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open depth raster: {depth_raster}")

    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    transform = dataset.GetGeoTransform()
    coord_transform = create_coordinate_transform(source_crs, dataset)

    xs = pd.to_numeric(df[x_column], errors="coerce")
    ys = pd.to_numeric(df[y_column], errors="coerce")
    depths = pd.Series(np.nan, index=df.index, dtype="float64")

    for index, lon, lat in zip(df.index, xs, ys):
        if pd.isna(lon) or pd.isna(lat):
            continue

        try:
            x, y, _ = coord_transform.TransformPoint(float(lon), float(lat))
            col = int((x - transform[0]) / transform[1])
            row = int((y - transform[3]) / transform[5])
            if not (0 <= col < dataset.RasterXSize and 0 <= row < dataset.RasterYSize):
                continue

            value = band.ReadAsArray(col, row, 1, 1)[0, 0]
            if not np.isfinite(value):
                continue
            if nodata is not None and np.isclose(value, nodata):
                continue
            depths.loc[index] = float(value)
        except (RuntimeError, TypeError, ValueError):
            continue

    dataset = None
    return depths


def add_depth_to_csv(
    input_file: Path,
    output_file: Path,
    depth_raster: Path,
    x_column: str,
    y_column: str,
    source_crs: str,
    depth_column: str,
) -> None:
    delimiter = sniff_delimiter(input_file)
    df = pd.read_csv(input_file, sep=delimiter, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    df[depth_column] = sample_depths(df, depth_raster, x_column, y_column, source_crs)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, sep=delimiter)

    valid_count = int(df[depth_column].notna().sum())
    print(
        f"Wrote {output_file} using {depth_raster.name} "
        f"({valid_count} sampled depths in {depth_column})"
    )


def find_batch_input(insitu_root: Path, config: DepthConfig) -> Path | None:
    campaign_dir = insitu_root / config.campaign_id / "preprocessing"
    if not campaign_dir.exists():
        return None

    candidates = sorted(
        path
        for path in campaign_dir.glob("*.csv")
        if "VIRB" in path.name.upper()
        and "RAMSES" in path.name.upper()
        and not path.stem.endswith("_depth")
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"Multiple candidate CSVs for {config.campaign_id}; using {candidates[0]}")
    return candidates[0]


def run_single(args: argparse.Namespace, configs: list[DepthConfig]) -> None:
    if args.input_file is None:
        raise ValueError("Pass --input-file or use --batch.")

    input_file = args.input_file.resolve()
    config = select_config(configs, input_file, args.campaign_date, args.campaign_code)
    depth_raster = args.depth_root.resolve() / config.output_name
    if not depth_raster.exists():
        raise FileNotFoundError(f"Depth raster does not exist: {depth_raster}")

    output_file = args.output_file.resolve() if args.output_file else input_file
    add_depth_to_csv(
        input_file=input_file,
        output_file=output_file,
        depth_raster=depth_raster,
        x_column=args.x_column,
        y_column=args.y_column,
        source_crs=args.source_crs,
        depth_column=args.depth_column,
    )


def run_batch(args: argparse.Namespace, configs: list[DepthConfig]) -> None:
    for config in configs:
        input_file = find_batch_input(args.insitu_root.resolve(), config)
        if input_file is None:
            print(f"Skipping {config.campaign_id}: no VIRB/RAMSES preprocessing CSV found")
            continue

        depth_raster = args.depth_root.resolve() / config.output_name
        if not depth_raster.exists():
            print(f"Skipping {config.campaign_id}: missing depth raster {depth_raster}")
            continue

        add_depth_to_csv(
            input_file=input_file,
            output_file=input_file,
            depth_raster=depth_raster,
            x_column=args.x_column,
            y_column=args.y_column,
            source_crs=args.source_crs,
            depth_column=args.depth_column,
        )


def main() -> None:
    args = parse_args()
    configs = read_depth_config(args.config.resolve())

    if args.batch:
        run_batch(args, configs)
    else:
        run_single(args, configs)


if __name__ == "__main__":
    main()
