"""
Create water-only lake depth rasters from bathymetry mosaics and binary masks.

Depth is calculated as lake_level - bottom_elevation. If lake_level is not
provided for a lake, it is inferred as the minimum valid masked bathymetry value
plus the configured maximum depth. The binary mask defines water; bathymetry
thresholds are only used to reject invalid depths and outliers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal


DEFAULT_BASE = Path(r"C:\Users\puehrifi\Documents\insitu\all\qgis_transect_bb")
DEFAULT_BATHY_ROOT = DEFAULT_BASE / "bathymetry_mosaics"
DEFAULT_MASK_ROOT = DEFAULT_BASE / "binary_masks"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE / "depth"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("lake_depth_config.csv")
DEFAULT_CRS = "EPSG:2056"
DEFAULT_NODATA = -9999.0


@dataclass(frozen=True)
class LakeConfig:
    lake: str
    campaign_date: str
    campaign_code: str
    bathy_name: str
    mask_name: str
    output_name: str
    max_depth: float
    lake_level: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create lake depth rasters using bathymetry mosaics and binary water masks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"CSV with lake levels, max depths, masks, and outputs. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--bathy-root",
        type=Path,
        default=DEFAULT_BATHY_ROOT,
        help=f"Folder containing bathymetry mosaics. Default: {DEFAULT_BATHY_ROOT}",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=DEFAULT_MASK_ROOT,
        help=f"Folder containing binary water masks. Default: {DEFAULT_MASK_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Folder for depth outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--lake",
        action="append",
        help="Lake to process. Repeat for multiple lakes. Default: all configured lakes.",
    )
    parser.add_argument(
        "--campaign-date",
        action="append",
        help="Campaign date in YYYYMMDD format. Repeat for multiple dates. Default: all dates.",
    )
    parser.add_argument(
        "--lake-level",
        type=float,
        help=(
            "Explicit water surface elevation for a single --lake run. "
            "Normally this comes from the config CSV."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing depth rasters and temporary masks.",
    )
    parser.add_argument(
        "--keep-mask",
        action="store_true",
        help="Keep the mask warped to the bathymetry grid.",
    )
    parser.add_argument(
        "--no-fill-missing-depths",
        action="store_true",
        help="Do not interpolate nodata holes that fall inside the binary water mask.",
    )
    parser.add_argument(
        "--fill-max-distance-pixels",
        type=float,
        default=2000.0,
        help="Maximum pixel search distance for filling internal depth gaps. Default: 2000.",
    )
    return parser.parse_args()


def read_lake_config(config_path: Path) -> list[LakeConfig]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config CSV does not exist: {config_path}")

    configs = []
    with config_path.open(newline="", encoding="utf-8") as config_file:
        reader = csv.DictReader(config_file)
        required = {
            "lake",
            "campaign_date",
            "campaign_code",
            "bathy_name",
            "mask_name",
            "lake_level_masl",
            "max_depth_m",
            "output_name",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing config columns: {', '.join(sorted(missing))}")

        for row in reader:
            configs.append(
                LakeConfig(
                    lake=row["lake"].strip().lower(),
                    campaign_date=row["campaign_date"].strip(),
                    campaign_code=row["campaign_code"].strip(),
                    bathy_name=row["bathy_name"].strip(),
                    mask_name=row["mask_name"].strip(),
                    output_name=row["output_name"].strip(),
                    max_depth=float(row["max_depth_m"]),
                    lake_level=float(row["lake_level_masl"]),
                )
            )

    if not configs:
        raise ValueError(f"No lake configurations found in {config_path}")
    return configs


def open_raster(path: Path) -> tuple[gdal.Dataset, float]:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")

    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = DEFAULT_NODATA
    return dataset, float(nodata)


def iter_windows(dataset: gdal.Dataset, block_size: int = 1024):
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    for yoff in range(0, height, block_size):
        ysize = min(block_size, height - yoff)
        for xoff in range(0, width, block_size):
            xsize = min(block_size, width - xoff)
            yield xoff, yoff, xsize, ysize


def warp_mask_to_bathy_grid(
    mask_path: Path,
    bathy_dataset: gdal.Dataset,
    warped_mask_path: Path,
    overwrite: bool,
) -> Path:
    if warped_mask_path.exists() and not overwrite:
        return warped_mask_path
    if warped_mask_path.exists():
        warped_mask_path.unlink()

    gt = bathy_dataset.GetGeoTransform()
    width = bathy_dataset.RasterXSize
    height = bathy_dataset.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + width * gt[1]
    ymin = ymax + height * gt[5]

    options = gdal.WarpOptions(
        dstSRS=DEFAULT_CRS,
        outputBounds=(xmin, ymin, xmax, ymax),
        width=width,
        height=height,
        resampleAlg="near",
        srcNodata=0,
        dstNodata=0,
        outputType=gdal.GDT_Byte,
        creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )
    warped = gdal.Warp(str(warped_mask_path), str(mask_path), options=options)
    if warped is None:
        raise RuntimeError(f"Failed to warp mask {mask_path}")
    warped.FlushCache()
    warped = None
    return warped_mask_path


def infer_lake_level(
    bathy_dataset: gdal.Dataset,
    mask_dataset: gdal.Dataset,
    bathy_nodata: float,
    max_depth: float,
) -> float:
    deepest_bottom = np.inf
    for xoff, yoff, xsize, ysize in iter_windows(bathy_dataset):
        bathy = bathy_dataset.GetRasterBand(1).ReadAsArray(xoff, yoff, xsize, ysize)
        mask = mask_dataset.GetRasterBand(1).ReadAsArray(xoff, yoff, xsize, ysize)
        valid_bathy = (mask == 1) & np.isfinite(bathy) & (bathy != bathy_nodata)
        if np.any(valid_bathy):
            deepest_bottom = min(deepest_bottom, float(np.nanmin(bathy[valid_bathy])))

    if not np.isfinite(deepest_bottom):
        raise ValueError("Cannot infer lake level because no valid masked bathymetry pixels exist.")
    return deepest_bottom + max_depth


def write_depth(
    output_path: Path,
    template_dataset: gdal.Dataset,
    mask_dataset: gdal.Dataset,
    bathy_nodata: float,
    max_lake_depth: float,
    lake_level: float,
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        print(f"Using existing {output_path}")
        return -1
    if output_path.exists():
        output_path.unlink()
    aux_path = output_path.with_name(f"{output_path.name}.aux.xml")
    if aux_path.exists():
        aux_path.unlink()

    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(output_path),
        template_dataset.RasterXSize,
        template_dataset.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )
    if output is None:
        raise RuntimeError(f"Failed to create {output_path}")

    output.SetGeoTransform(template_dataset.GetGeoTransform())
    output.SetProjection(template_dataset.GetProjection())
    band = output.GetRasterBand(1)
    band.SetNoDataValue(DEFAULT_NODATA)

    valid_count = 0
    bathy_band = template_dataset.GetRasterBand(1)
    mask_band = mask_dataset.GetRasterBand(1)
    for xoff, yoff, xsize, ysize in iter_windows(template_dataset):
        bathy = bathy_band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
        mask = mask_band.ReadAsArray(xoff, yoff, xsize, ysize)
        water = mask == 1
        valid_bathy = water & np.isfinite(bathy) & (bathy != bathy_nodata)
        depth_candidate = lake_level - bathy
        valid_depth = (
            valid_bathy
            & np.isfinite(depth_candidate)
            & (depth_candidate > 0)
            & (depth_candidate <= max_lake_depth)
        )
        depth = np.where(valid_depth, depth_candidate, DEFAULT_NODATA).astype(np.float32)
        band.WriteArray(depth, xoff, yoff)
        valid_count += int(np.count_nonzero(valid_depth))

    band.FlushCache()
    output.FlushCache()
    output = None
    return valid_count


def remask_depth_to_water(
    output_path: Path,
    mask_dataset: gdal.Dataset,
    max_lake_depth: float,
) -> int:
    dataset = gdal.Open(str(output_path), gdal.GA_Update)
    if dataset is None:
        raise FileNotFoundError(f"Cannot reopen depth raster for masking: {output_path}")

    depth_band = dataset.GetRasterBand(1)
    mask_band = mask_dataset.GetRasterBand(1)
    valid_count = 0

    for xoff, yoff, xsize, ysize in iter_windows(dataset):
        depth = depth_band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
        mask = mask_band.ReadAsArray(xoff, yoff, xsize, ysize)
        valid_depth = (
            (mask == 1)
            & np.isfinite(depth)
            & (depth != DEFAULT_NODATA)
            & (depth > 0)
            & (depth <= max_lake_depth)
        )
        depth = np.where(valid_depth, depth, DEFAULT_NODATA).astype(np.float32)
        depth_band.WriteArray(depth, xoff, yoff)
        valid_count += int(np.count_nonzero(valid_depth))

    depth_band.FlushCache()
    dataset.FlushCache()
    dataset = None
    return valid_count


def fill_internal_depth_gaps(
    output_path: Path,
    mask_dataset: gdal.Dataset,
    max_lake_depth: float,
    max_distance_pixels: float,
) -> int:
    dataset = gdal.Open(str(output_path), gdal.GA_Update)
    if dataset is None:
        raise FileNotFoundError(f"Cannot reopen depth raster for filling: {output_path}")

    band = dataset.GetRasterBand(1)
    mask_band = mask_dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    source_mask_dataset = gdal.GetDriverByName("MEM").Create(
        "",
        dataset.RasterXSize,
        dataset.RasterYSize,
        1,
        gdal.GDT_Byte,
    )
    source_mask_band = source_mask_dataset.GetRasterBand(1)

    # FillNodata uses non-zero mask pixels as interpolation sources. Keep
    # sources limited to valid water depths so land values cannot bias the fill.
    for xoff, yoff, xsize, ysize in iter_windows(dataset):
        depth = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
        mask = mask_band.ReadAsArray(xoff, yoff, xsize, ysize)
        valid_source = (
            (mask == 1)
            & np.isfinite(depth)
            & (depth != nodata)
            & (depth > 0)
            & (depth <= max_lake_depth)
        )
        source_mask_band.WriteArray(valid_source.astype(np.uint8), xoff, yoff)
    source_mask_band.FlushCache()

    result = gdal.FillNodata(
        targetBand=band,
        maskBand=source_mask_band,
        maxSearchDist=min(max_distance_pixels, 100.0),
        smoothingIterations=0,
    )
    band.FlushCache()
    dataset.FlushCache()
    source_mask_dataset = None
    dataset = None

    if result != 0:
        raise RuntimeError(f"GDAL FillNodata failed for {output_path} with code {result}")
    remask_depth_to_water(output_path, mask_dataset, max_lake_depth)
    return fill_remaining_depth_gaps_nearest(
        output_path,
        mask_dataset,
        max_lake_depth,
        max_distance_pixels,
    )


def fill_remaining_depth_gaps_nearest(
    output_path: Path,
    mask_dataset: gdal.Dataset,
    max_lake_depth: float,
    max_distance_pixels: float,
) -> int:
    from scipy import ndimage

    dataset = gdal.Open(str(output_path), gdal.GA_Update)
    if dataset is None:
        raise FileNotFoundError(f"Cannot reopen depth raster for nearest fill: {output_path}")

    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    mask_band = mask_dataset.GetRasterBand(1)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    missing = np.zeros((height, width), dtype=bool)

    for xoff, yoff, xsize, ysize in iter_windows(dataset):
        depth = band.ReadAsArray(xoff, yoff, xsize, ysize)
        mask = mask_band.ReadAsArray(xoff, yoff, xsize, ysize)
        valid_depth = (
            (mask == 1)
            & np.isfinite(depth)
            & (depth != nodata)
            & (depth > 0)
            & (depth <= max_lake_depth)
        )
        missing[yoff : yoff + ysize, xoff : xoff + xsize] = (mask == 1) & ~valid_depth

    if not missing.any():
        dataset = None
        return remask_depth_to_water(output_path, mask_dataset, max_lake_depth)

    labels, component_count = ndimage.label(missing)
    objects = ndimage.find_objects(labels)
    max_distance = max(1, int(round(max_distance_pixels)))
    initial_padding = min(256, max_distance)

    for component_id, component_slice in enumerate(objects, start=1):
        if component_slice is None:
            continue

        y_slice, x_slice = component_slice
        padding = initial_padding
        while True:
            x0 = max(0, x_slice.start - padding)
            x1 = min(width, x_slice.stop + padding)
            y0 = max(0, y_slice.start - padding)
            y1 = min(height, y_slice.stop + padding)
            xsize = x1 - x0
            ysize = y1 - y0

            depth = band.ReadAsArray(x0, y0, xsize, ysize).astype(np.float32)
            mask = mask_band.ReadAsArray(x0, y0, xsize, ysize)
            valid_source = (
                (mask == 1)
                & np.isfinite(depth)
                & (depth != nodata)
                & (depth > 0)
                & (depth <= max_lake_depth)
            )
            if valid_source.any() or padding >= max_distance:
                break
            padding = min(max_distance, padding * 2)

        if not valid_source.any():
            continue

        local_labels = labels[y0:y1, x0:x1]
        component = local_labels == component_id
        distances, indices = ndimage.distance_transform_edt(
            ~valid_source,
            return_indices=True,
        )
        fill_target = component & (distances <= max_distance)
        if not fill_target.any():
            continue

        depth[fill_target] = depth[indices[0][fill_target], indices[1][fill_target]]
        band.WriteArray(depth, x0, y0)

    band.FlushCache()
    dataset.FlushCache()
    dataset = None
    return remask_depth_to_water(output_path, mask_dataset, max_lake_depth)


def create_lake_depth_raster(
    bathy_raster_path: Path,
    mask_raster_path: Path,
    output_raster_path: Path,
    max_lake_depth: float,
    lake_level: float | None = None,
    overwrite: bool = False,
    keep_mask: bool = False,
    fill_missing_depths: bool = True,
    fill_max_distance_pixels: float = 2000.0,
) -> Path:
    output_raster_path.parent.mkdir(parents=True, exist_ok=True)
    warped_mask_path = output_raster_path.with_name(
        f"{output_raster_path.stem}_water_mask_on_bathy_grid.tif"
    )

    bathy_dataset, bathy_nodata = open_raster(bathy_raster_path)
    warp_mask_to_bathy_grid(mask_raster_path, bathy_dataset, warped_mask_path, overwrite)
    mask_dataset, _ = open_raster(warped_mask_path)

    if lake_level is None:
        lake_level = infer_lake_level(bathy_dataset, mask_dataset, bathy_nodata, max_lake_depth)

    valid_count = write_depth(
        output_raster_path,
        bathy_dataset,
        mask_dataset,
        bathy_nodata,
        max_lake_depth,
        lake_level,
        overwrite,
    )
    if fill_missing_depths:
        valid_count = fill_internal_depth_gaps(
            output_raster_path,
            mask_dataset,
            max_lake_depth,
            fill_max_distance_pixels,
        )
    mask_dataset = None
    bathy_dataset = None

    if not keep_mask and warped_mask_path.exists():
        warped_mask_path.unlink()

    print(
        f"Created {output_raster_path} "
        f"(lake_level={lake_level:.3f}, max_depth={max_lake_depth:g}, "
        f"valid_pixels={valid_count}, fill_missing_depths={fill_missing_depths})"
    )
    return output_raster_path


def selected_configs(args: argparse.Namespace, configs: list[LakeConfig]) -> list[LakeConfig]:
    if args.lake_level is not None and (not args.lake or len(args.lake) != 1):
        raise ValueError("--lake-level can only be used with exactly one --lake.")

    selected = configs
    if args.lake:
        lakes = {lake.lower() for lake in args.lake}
        selected = [config for config in selected if config.lake in lakes]
        missing = lakes - {config.lake for config in selected}
        if missing:
            raise ValueError(f"No config rows found for lake(s): {', '.join(sorted(missing))}")

    if args.campaign_date:
        dates = set(args.campaign_date)
        selected = [config for config in selected if config.campaign_date in dates]
        missing = dates - {config.campaign_date for config in selected}
        if missing:
            raise ValueError(f"No config rows found for campaign date(s): {', '.join(sorted(missing))}")

    if args.lake_level is not None:
        selected = [
            LakeConfig(
                lake=config.lake,
                campaign_date=config.campaign_date,
                campaign_code=config.campaign_code,
                bathy_name=config.bathy_name,
                mask_name=config.mask_name,
                output_name=config.output_name,
                max_depth=config.max_depth,
                lake_level=args.lake_level,
            )
            for config in selected
        ]

    if not selected:
        raise ValueError("No lake configurations selected.")
    return selected


def main() -> None:
    args = parse_args()
    bathy_root = args.bathy_root.resolve()
    mask_root = args.mask_root.resolve()
    output_root = args.output_root.resolve()
    configs = read_lake_config(args.config.resolve())

    outputs = []
    for config in selected_configs(args, configs):
        bathy_path = bathy_root / config.bathy_name
        mask_path = mask_root / config.mask_name
        output_path = output_root / config.output_name

        if not bathy_path.exists():
            raise FileNotFoundError(f"Missing bathymetry mosaic for {config.lake}: {bathy_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing binary mask for {config.lake}: {mask_path}")

        outputs.append(
            create_lake_depth_raster(
                bathy_raster_path=bathy_path,
                mask_raster_path=mask_path,
                output_raster_path=output_path,
                max_lake_depth=config.max_depth,
                lake_level=config.lake_level,
                overwrite=args.overwrite,
                keep_mask=args.keep_mask,
                fill_missing_depths=not args.no_fill_missing_depths,
                fill_max_distance_pixels=args.fill_max_distance_pixels,
            )
        )

    print("\nDone. Depth rasters:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
