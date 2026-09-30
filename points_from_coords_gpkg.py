import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def read_csv_robust(path):
    attempts = [
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": "\t", "decimal": "."},
        {"sep": "\t", "decimal": ","},
    ]

    for opts in attempts:
        try:
            df = pd.read_csv(path, **opts)

            # Successful parse should usually produce more than 1 column.
            if len(df.columns) > 1:
                df.columns = df.columns.str.strip()
                print(f"  OK CSV read successfully with options: {opts}")
                print(f"  OK Columns detected: {df.columns.tolist()}")
                return df

        except Exception:
            continue

    raise ValueError(f"Could not read CSV correctly: {path}")


def csv_to_gpkg(csv_file, output_dir=None):
    csv_file = Path(csv_file)
    print(f"Processing {csv_file.name}...")

    try:
        df = read_csv_robust(csv_file)
    except ValueError as e:
        print(f"  WARNING {e}")
        return None

    # --- Step 1: Fold relative azimuth angle if column exists ---
    # for col in df.columns:
    #     if col.lower() == "relative azimuth angle":
    #         df[col] = df[col] % 360
    #         df[col] = np.where(df[col] > 180, 360 - df[col], df[col])
    #         print(f"  OK Updated '{col}' column with symmetric angles.")
    #         break

    # --- Step 2: Create geometry column ---
    coordinate_pairs = [
        ("x-coordinate", "y-coordinate"),
        ("insitu_lon", "insitu_lat"),
        ("eo_lon", "eo_lat"),
    ]
    xy_columns = next(
        (
            (x_col, y_col)
            for x_col, y_col in coordinate_pairs
            if {x_col, y_col}.issubset(df.columns)
        ),
        None,
    )

    if xy_columns is None:
        print(f"  WARNING Skipping {csv_file.name} (missing coordinate columns).")
        return None

    x_col, y_col = xy_columns
    print(f"  OK Using coordinate columns: {x_col}, {y_col}")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    df = df.dropna(subset=[x_col, y_col]).copy()

    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Input: 20230610_CST_metadata_with_Rrs_conv_L8_qc.csv
    # Output: 20230610_CST_transect_qc.gpkg
    parts = csv_file.stem.split("_")
    if len(parts) >= 2:
        out_name = f"{parts[0]}_{parts[1]}_transect_qc.gpkg"
    else:
        out_name = csv_file.stem + "_transect_qc.gpkg"

    output_dir = Path(output_dir) if output_dir else csv_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg = output_dir / out_name

    gdf.to_file(output_gpkg, layer="points_layer", driver="GPKG")
    print(f"  OK Saved GeoPackage to {output_gpkg}")
    return output_gpkg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a point GeoPackage from CSV x-coordinate/y-coordinate columns."
    )
    parser.add_argument("csv_files", nargs="*", type=Path, help="Input CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for output GeoPackages. Defaults to each CSV's parent directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # List of input CSV files used when none are provided on the command line.
    csv_files = args.csv_files or [
        Path(
            r"C:\Users\puehrifi\Documents\insitu\20231008_ZRH\visualization_analysis\quality_control\20231008_ZRH_metadata_with_Rrs_S2B_final_qc.csv"
        ),
        # Path("/Users/fionapuhringer/Documents/Geographie/Master/msc_thesis/adjacency/data_quality_control/previous_campaigns/20230610_CST_filtered/20230610_CST_metadata_with_Rrs_conv_L8.csv"),
        # Path("/Users/fionapuhringer/Documents/Geographie/Master/msc_thesis/adjacency/data_quality_control/previous_campaigns/20231008_ZRH/20231008_ZRH_metadata_with_Rrs_conv_L9.csv"),
        # Path("/Users/fionapuhringer/Documents/Geographie/Master/msc_thesis/adjacency/data_quality_control/previous_campaigns/20240618_BIE/20240618_BIE_metadata_with_Rrs_conv_L9.csv"),
        # Path("/Users/fionapuhringer/Documents/Geographie/Master/msc_thesis/adjacency/data_quality_control/previous_campaigns/20230610_CST_filtered/20230610_CST_metadata_with_Rrs_L8.csv")
    ]

    for csv_file in csv_files:
        csv_to_gpkg(csv_file, args.output_dir)


if __name__ == "__main__":
    main()
