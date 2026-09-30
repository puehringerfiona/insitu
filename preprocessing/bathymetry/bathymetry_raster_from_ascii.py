"""
Build bathymetry GeoTIFF mosaics from swissBATHY3D ESRI ASCII tiles.

The working output stays in EPSG:2056 (CH1903+ / LV95), so downstream depth
and distance calculations use meter-based pixels. Reproject to EPSG:4326 only
as a separate display/export step if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from osgeo import gdal


DEFAULT_ASCII_ROOT = Path(
    r"C:\Users\puehrifi\Documents\insitu\all\qgis_transect_bb\ascii_lake_bathy"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ASCII_ROOT.parent / "bathymetry_mosaics"
DEFAULT_CRS = "EPSG:2056"
DEFAULT_NODATA = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EPSG:2056 bathymetry mosaics from lake ASCII tile folders."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_ASCII_ROOT,
        help=f"Folder containing one subfolder per lake. Default: {DEFAULT_ASCII_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output folder for mosaics. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--lake",
        action="append",
        help="Lake subfolder name to process. Repeat for multiple lakes. Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mosaics and VRTs.",
    )
    parser.add_argument(
        "--keep-vrt",
        action="store_true",
        help="Keep the intermediate VRT next to the output GeoTIFF.",
    )
    return parser.parse_args()


def build_lake_mosaic(
    lake_dir: Path,
    output_root: Path,
    overwrite: bool = False,
    keep_vrt: bool = False,
) -> Path:
    ascii_files = sorted(lake_dir.glob("*.asc"))
    if not ascii_files:
        raise FileNotFoundError(f"No .asc files found in {lake_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    vrt_path = output_root / f"{lake_dir.name}_bathy_2056.vrt"
    output_path = output_root / f"{lake_dir.name}_bathy_2056.tif"

    if output_path.exists() and not overwrite:
        print(f"Using existing {output_path}")
        return output_path

    if vrt_path.exists():
        vrt_path.unlink()
    if output_path.exists():
        output_path.unlink()

    print(f"Building {lake_dir.name}: {len(ascii_files)} ASCII tiles")
    vrt_options = gdal.BuildVRTOptions(
        srcNodata=DEFAULT_NODATA,
        VRTNodata=DEFAULT_NODATA,
        outputSRS=DEFAULT_CRS,
    )
    vrt = gdal.BuildVRT(str(vrt_path), [str(path) for path in ascii_files], options=vrt_options)
    if vrt is None:
        raise RuntimeError(f"Failed to build VRT for {lake_dir}")
    vrt.FlushCache()
    vrt = None

    translate_options = gdal.TranslateOptions(
        noData=DEFAULT_NODATA,
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )
    output = gdal.Translate(str(output_path), str(vrt_path), options=translate_options)
    if output is None:
        raise RuntimeError(f"Failed to create {output_path}")
    output.FlushCache()
    output = None

    if not keep_vrt and vrt_path.exists():
        vrt_path.unlink()

    print(f"Created {output_path}")
    return output_path


def find_lake_dirs(input_root: Path, requested_lakes: list[str] | None) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    lake_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if requested_lakes:
        requested = set(requested_lakes)
        lake_dirs = [path for path in lake_dirs if path.name in requested]
        missing = requested - {path.name for path in lake_dirs}
        if missing:
            raise FileNotFoundError(f"Lake subfolders not found: {', '.join(sorted(missing))}")

    if not lake_dirs:
        raise FileNotFoundError(f"No lake subfolders found in {input_root}")
    return lake_dirs


def main() -> None:
    args = parse_args()
    outputs = [
        build_lake_mosaic(
            lake_dir=lake_dir,
            output_root=args.output_root.resolve(),
            overwrite=args.overwrite,
            keep_vrt=args.keep_vrt,
        )
        for lake_dir in find_lake_dirs(args.input_root.resolve(), args.lake)
    ]

    print("\nDone. Bathymetry mosaics:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
