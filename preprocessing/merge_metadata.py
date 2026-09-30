import pandas as pd
import glob
import os
import re
import pvlib
from geographiclib.geodesic import Geodesic
import numpy as np
import pyproj

def fill_closest_filename(df):
    """Fill missing source_filename from nearest existing timestamp."""
    df = df.copy()
    df["datetime (UTC)"] = pd.to_datetime(
        df["datetime (UTC)"],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    known = df[["datetime (UTC)", "source_filename"]].dropna(subset=["source_filename"]).copy()
    missing_mask = df["source_filename"].isna() & df["datetime (UTC)"].notna()

    if known.empty or not missing_mask.any():
        return df

    known = known.sort_values("datetime (UTC)")
    missing = df.loc[missing_mask, ["datetime (UTC)"]].sort_values("datetime (UTC)").copy()

    nearest = pd.merge_asof(
        missing,
        known,
        on="datetime (UTC)",
        direction="nearest"
    )

    df.loc[missing.index, "source_filename"] = nearest["source_filename"].values
    return df


def ensure_utc(series, local_tz="Europe/Zurich"):
    """
    Ensure a datetime series is timezone-aware UTC.
    If naive -> localize to local_tz, then convert to UTC.
    If already tz-aware -> convert to UTC.
    """
    dt = pd.to_datetime(series, errors="coerce", format="mixed")

    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")

    return dt


def calculate_heading(lat1, lon1, lat2, lon2):
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    return (fwd_azimuth + 360) % 360


def compute_heading_for_mask(df, target_mask):
    """
    Compute heading only for rows where target_mask is True.
    Uses previous and next valid coordinates in the full dataframe.
    """
    df = df.copy()
    if "heading" not in df.columns:
        df["heading"] = np.nan

    coords_valid = df["x-coordinate"].notna() & df["y-coordinate"].notna()

    for idx in df.index[target_mask]:
        prev_candidates = df.index[(df.index < idx) & coords_valid]
        next_candidates = df.index[(df.index > idx) & coords_valid]

        if len(prev_candidates) == 0 or len(next_candidates) == 0:
            continue

        prev_idx = prev_candidates[-1]
        next_idx = next_candidates[0]

        lat1, lon1 = df.at[prev_idx, "y-coordinate"], df.at[prev_idx, "x-coordinate"]
        lat2, lon2 = df.at[next_idx, "y-coordinate"], df.at[next_idx, "x-coordinate"]

        if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2):
            df.at[idx, "heading"] = calculate_heading(lat1, lon1, lat2, lon2)

    return df


def compute_solar_for_mask(df, target_mask):
    """
    Compute solar quantities only for rows where target_mask is True.
    """
    df = df.copy()

    for col in [
        "instantaneous solar azimuth angle",
        "instantaneous solar zenith angle",
        "relative azimuth angle",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    valid_mask = (
        target_mask
        & df["datetime (UTC)"].notna()
        & df["y-coordinate"].notna()
        & df["x-coordinate"].notna()
        & df["heading"].notna()
    )

    if valid_mask.any():
        solar_position = pvlib.solarposition.get_solarposition(
            df.loc[valid_mask, "datetime (UTC)"],
            df.loc[valid_mask, "y-coordinate"],
            df.loc[valid_mask, "x-coordinate"],
        )

        df.loc[valid_mask, "instantaneous solar azimuth angle"] = solar_position["azimuth"].values
        df.loc[valid_mask, "instantaneous solar zenith angle"] = solar_position["zenith"].values
        df.loc[valid_mask, "relative azimuth angle"] = (
            df.loc[valid_mask, "instantaneous solar azimuth angle"]
            - df.loc[valid_mask, "heading"]
        ) % 360

    return df


def compute_speed_for_mask(df, target_mask):
    """
    Compute speed only for rows where target_mask is True.
    Uses previous and next valid coordinates/timestamps in the full dataframe.
    """
    df = df.copy()
    df["datetime (UTC)"] = pd.to_datetime(
        df["datetime (UTC)"],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    if "speed" not in df.columns:
        df["speed"] = np.nan

    valid = (
        df["datetime (UTC)"].notna()
        & df["x-coordinate"].notna()
        & df["y-coordinate"].notna()
    )

    geod = Geodesic.WGS84

    for idx in df.index[target_mask]:
        prev_candidates = df.index[(df.index < idx) & valid]
        next_candidates = df.index[(df.index > idx) & valid]

        if len(prev_candidates) == 0 or len(next_candidates) == 0:
            continue

        prev_idx = prev_candidates[-1]
        next_idx = next_candidates[0]

        lat1, lon1 = df.at[prev_idx, "y-coordinate"], df.at[prev_idx, "x-coordinate"]
        lat2, lon2 = df.at[next_idx, "y-coordinate"], df.at[next_idx, "x-coordinate"]
        t1 = df.at[prev_idx, "datetime (UTC)"]
        t2 = df.at[next_idx, "datetime (UTC)"]

        if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2) and pd.notna(t1) and pd.notna(t2):
            time_diff = (t2 - t1).total_seconds()
            if time_diff > 0:
                g = geod.Inverse(lat1, lon1, lat2, lon2)
                distance = g["s12"]
                df.at[idx, "speed"] = distance / time_diff

    return df


# ========== LOAD FILES ==========

file_pattern_virb = r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\metadata\20240619_WAL\20240619_WAL_VIRB.csv"
file_list_virb = glob.glob(file_pattern_virb)

inclination_file = r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\metadata\20240619_WAL\20240619_WAL_ramses_metadata.csv"

df_inclination = pd.read_csv(inclination_file)
df_virb = pd.read_csv(file_list_virb[0], sep=";")

# ========== STANDARDIZE COLUMN NAMES ==========

# VIRB file already has:
# source_filename,datetime (Europe/Zurich),datetime (UTC),x-coordinate,y-coordinate,heading,
# instantaneous solar azimuth angle,instantaneous solar zenith angle,relative azimuth angle,speed,
# heading_fit,enhanced_altitude,enhanced_speed,velocity

# RAMSES file has:
# filename,DateTime,Incl_X,Incl_Y,Incl_V

# Rename RAMSES filename to avoid confusion with VIRB source_filename
if "filename" in df_inclination.columns:
    df_inclination = df_inclination.rename(columns={"filename": "filename_ramses"})

# ========== DATETIME HANDLING ==========

# VIRB
virb_local_col = "datetime (Europe/Zurich)"
if virb_local_col in df_virb.columns:
    df_virb["datetime (UTC)"] = ensure_utc(df_virb[virb_local_col], local_tz="Europe/Zurich")
elif "datetime (UTC)" in df_virb.columns:
    df_virb["datetime (UTC)"] = ensure_utc(df_virb["datetime (UTC)"], local_tz="UTC")
else:
    raise ValueError("VIRB file must contain either 'datetime (Europe/Zurich)' or 'datetime (UTC)'.")

df_virb["datetime (Europe/Zurich)"] = df_virb["datetime (UTC)"].dt.tz_convert("Europe/Zurich")

# RAMSES
df_inclination["datetime (UTC)"] = ensure_utc(df_inclination["DateTime"], local_tz="Europe/Zurich")
df_inclination["datetime (Europe/Zurich)"] = df_inclination["datetime (UTC)"].dt.tz_convert("Europe/Zurich")

# Extract fid_ramses from RAMSES filename if available
if "filename_ramses" in df_inclination.columns:
    df_inclination["fid_ramses"] = df_inclination["filename_ramses"].apply(
        lambda x: re.search(r"_(\d{5})\.dat$", x).group(1)
        if pd.notna(x) and re.search(r"_(\d{5})\.dat$", x)
        else None
    )

# ========== KEEP TRACK OF DATA ORIGIN ==========

df_virb["is_virb_row"] = True
df_inclination["is_virb_row"] = False

# ========== MERGE (FULL UNION) ==========

all_times = pd.to_datetime(
    pd.concat([df_virb["datetime (UTC)"], df_inclination["datetime (UTC)"]]),
    utc=True
).dropna().drop_duplicates().sort_values()

df_virb = df_virb.set_index("datetime (UTC)")
df_inclination = df_inclination.set_index("datetime (UTC)")

merged_df = pd.DataFrame(index=all_times)
merged_df = merged_df.join(df_virb, how="left")
merged_df = merged_df.join(df_inclination, how="left", rsuffix="_ramses")

merged_df = merged_df.reset_index().rename(columns={"index": "datetime (UTC)"})
merged_df["datetime (Europe/Zurich)"] = merged_df["datetime (UTC)"].dt.tz_convert("Europe/Zurich")

# Mark rows that were introduced from RAMSES only
merged_df["is_ramses_only_row"] = merged_df["is_virb_row"].isna() & merged_df["fid_ramses"].notna()

# Unix timestamp
merged_df["timestamp (UTC)"] = merged_df["datetime (UTC)"].apply(
    lambda x: int(x.timestamp()) if pd.notna(x) else np.nan
)

# ========== INTERPOLATION FOR RAMSES-ONLY ROWS ==========

if merged_df.index.name != "datetime (UTC)":
    merged_df = merged_df.set_index("datetime (UTC)")

for col in ["x-coordinate", "y-coordinate"]:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].interpolate(method="time")

for col in ["Incl_V", "Incl_X", "Incl_Y"]:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].interpolate(method="time")

merged_df = merged_df.reset_index()

# Fill VIRB source_filename onto nearest RAMSES-only rows
merged_df = fill_closest_filename(merged_df)

# Drop duplicate Unix timestamps if needed
merged_df = merged_df.drop_duplicates(subset="timestamp (UTC)", keep="first")

# Forward-fill RAMSES IDs/names
if "fid_ramses" in merged_df.columns:
    merged_df["fid_ramses"] = merged_df["fid_ramses"].ffill()
if "filename_ramses" in merged_df.columns:
    merged_df["filename_ramses"] = merged_df["filename_ramses"].ffill()

# ========== RE-COMPUTE HEADING, SPEED, SOLAR GEOMETRY ==========

target_mask = merged_df["is_ramses_only_row"].fillna(False)

merged_df = compute_heading_for_mask(merged_df, target_mask)
merged_df = compute_speed_for_mask(merged_df, target_mask)
merged_df = compute_solar_for_mask(merged_df, target_mask)

# ========== FINAL COLUMN ORDER ==========

final_columns = [
    "source_filename",
    "filename_ramses",
    "fid_ramses",
    "timestamp (UTC)",
    "datetime (UTC)",
    "datetime (Europe/Zurich)",
    "x-coordinate",
    "y-coordinate",
    "heading",
    "instantaneous solar azimuth angle",
    "instantaneous solar zenith angle",
    "relative azimuth angle",
    "speed",
    "Incl_V",
    "Incl_X",
    "Incl_Y",
]

merged_df = merged_df[[col for col in final_columns if col in merged_df.columns]]

# ========== EXPORT ==========

output_path = r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\metadata\20260226_ZRH\20260226_ZRH_VIRB_RAMSES_merged.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_csv(output_path, index=False)
print(f"Saved merged file to: {output_path}")
