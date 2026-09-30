"""
Extract metadata (lat, lon, datetime UTC) and calculate:
- heading
- cruising speed
- instantaneous solar azimuth angle
- instantaneous solar zenith angle
- relative solar azimuth angle

Supported inputs:
1) VIRB image folder (jpg/jpeg)
2) GPX file
3) FIT file

The rest of the logic stays the same:
extract -> calculate -> return/save CSV
"""

import os
import glob
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import piexif
from PIL import Image

import pvlib
import pyproj
from geographiclib.geodesic import Geodesic

import gpxpy
from fitparse import FitFile


# ========== CONFIG ==========

# Choose one:
#INPUT_TYPE = "virb"
INPUT_TYPE = "gpx"
# INPUT_TYPE = "fit"


# Input paths
IMAGE_DIR = r"C:\Users\puehrifi\Documents\campaigns\20260226_ZRH\Fotos\*"
GPX_FILE = r"C:\Users\puehrifi\Documents\campaigns\20230610_CST\insitu_data\t269605391_cst adjacency 10 june.gpx"
FIT_FILE = r"C:\Users\puehrifi\Documents\campaigns\20240618_BIE\insitu_data\2024-06-18-11-55-01.fit"

# Output path
OUTPUT_CSV = r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\metadata\20230610_CST\20230610_CST_VIRB.csv"

# Timezone used for VIRB EXIF timestamps
LOCAL_TIMEZONE = "Europe/Zurich"


# ========== FUNCTIONS FOR VIRB METADATA EXTRACTION ==========

def dms_to_decimal(dms, ref):
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60.0
    seconds = dms[2][0] / dms[2][1] / 3600.0
    decimal = degrees + minutes + seconds
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


def get_lat_lon_from_image(image_path):
    with Image.open(image_path) as image:
        exif_bytes = image.info.get("exif")
        if not exif_bytes:
            return None, None

        exif_data = piexif.load(exif_bytes)
        gps_info = exif_data.get("GPS", {})

        if gps_info:
            gps_latitude = gps_info.get(piexif.GPSIFD.GPSLatitude)
            gps_latitude_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef)
            gps_longitude = gps_info.get(piexif.GPSIFD.GPSLongitude)
            gps_longitude_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef)

            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = dms_to_decimal(gps_latitude, gps_latitude_ref.decode("utf-8"))
                lon = dms_to_decimal(gps_longitude, gps_longitude_ref.decode("utf-8"))
                return lat, lon

        return None, None


def get_datetime_from_exif(image_path, local_timezone=LOCAL_TIMEZONE):
    with Image.open(image_path) as image:
        exif_bytes = image.info.get("exif")
        if not exif_bytes:
            return None, None

        exif_data = piexif.load(exif_bytes)
        datetime_str = exif_data["Exif"].get(piexif.ExifIFD.DateTimeOriginal)

        if datetime_str:
            dt_naive = datetime.strptime(datetime_str.decode("utf-8"), "%Y:%m:%d %H:%M:%S")
            dt_local = dt_naive.replace(tzinfo=ZoneInfo(local_timezone))
            dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
            return dt_local, dt_utc

        return None, None


def extract_from_virb(image_dir):
    file_list = glob.glob(image_dir)

    file_names, latitudes, longitudes, timestamps_utc, timestamps_local = [], [], [], [], []

    for file in file_list:
        if file.lower().endswith((".jpg", ".jpeg")):
            try:
                file_name = os.path.basename(file)
                lat, lon = get_lat_lon_from_image(file)
                dt_local, dt_utc = get_datetime_from_exif(file)

                if lat is not None and lon is not None and dt_local and dt_utc:
                    file_names.append(file_name)
                    latitudes.append(lat)
                    longitudes.append(lon)
                    timestamps_local.append(dt_local)
                    timestamps_utc.append(dt_utc)
                else:
                    print(f"Skipping {file}: Missing GPS or DateTimeOriginal.")

            except Exception as e:
                print(f"Error processing {file}: {e}")

    df = pd.DataFrame({
        "source_filename": file_names,
        "x-coordinate": longitudes,
        "y-coordinate": latitudes,
        "datetime (UTC)": timestamps_utc,
        f"datetime ({LOCAL_TIMEZONE})": timestamps_local
    })

    return df


# ========== FUNCTIONS FOR GPX EXTRACTION ==========

def extract_from_gpx(gpx_file):
    with open(gpx_file, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    data = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                dt_utc = pd.to_datetime(point.time, utc=True, errors="coerce")
                dt_local = dt_utc.tz_convert(ZoneInfo(LOCAL_TIMEZONE)) if pd.notna(dt_utc) else pd.NaT

                data.append({
                    "source_filename": os.path.basename(gpx_file),
                    "x-coordinate": point.longitude,
                    "y-coordinate": point.latitude,
                    "datetime (UTC)": dt_utc,
                    f"datetime ({LOCAL_TIMEZONE})": dt_local
                })

    return pd.DataFrame(data)


# ========== FUNCTIONS FOR FIT EXTRACTION ==========

def semicircles_to_degrees(value):
    if value is None or pd.isna(value):
        return np.nan
    return value * (180.0 / 2**31)


def extract_from_fit(fit_file):
    """
    For this VIRB FIT structure, gps_metadata is the correct source.
    It contains:
    - timestamp
    - timestamp_ms
    - position_lat / position_long (already in degrees in this file)
    - enhanced_altitude
    - enhanced_speed
    - heading
    - utc_timestamp (absolute UTC, second resolution)
    - velocity

    We reconstruct high-resolution UTC timestamps from:
    rel_time_s = timestamp + timestamp_ms/1000
    anchored on the first valid utc_timestamp.
    """
    fitfile = FitFile(fit_file)
    rows = []

    for msg in fitfile.get_messages("gps_metadata"):
        row = {field.name: field.value for field in msg}
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No gps_metadata messages found in FIT file.")
        return pd.DataFrame()

    required_cols = ["timestamp", "timestamp_ms", "position_lat", "position_long", "utc_timestamp"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required FIT gps_metadata column: {col}")

    df = df.copy()

    # Parse numeric/time fields
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["position_lat"] = pd.to_numeric(df["position_lat"], errors="coerce")
    df["position_long"] = pd.to_numeric(df["position_long"], errors="coerce")
    df["enhanced_altitude"] = pd.to_numeric(df.get("enhanced_altitude", np.nan), errors="coerce")
    df["enhanced_speed"] = pd.to_numeric(df.get("enhanced_speed", np.nan), errors="coerce")
    df["heading_fit"] = pd.to_numeric(df.get("heading", np.nan), errors="coerce")
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")

    # Handle coordinates:
    # In your gps_metadata export they are already in degrees, but this keeps it robust.
    if df["position_lat"].dropna().abs().max() > 180:
        df["position_lat"] = df["position_lat"].apply(semicircles_to_degrees)
    if df["position_long"].dropna().abs().max() > 180:
        df["position_long"] = df["position_long"].apply(semicircles_to_degrees)

    # Build high-resolution relative time
    df["rel_time_s"] = df["timestamp"] + (df["timestamp_ms"] / 1000.0)

    first_idx = df["utc_timestamp"].first_valid_index()
    if first_idx is None:
        raise ValueError("No valid utc_timestamp found in gps_metadata.")

    base_utc = df.loc[first_idx, "utc_timestamp"]
    base_rel = df.loc[first_idx, "rel_time_s"]

    # Reconstruct precise UTC timestamps
    df["datetime (UTC)"] = base_utc + pd.to_timedelta(df["rel_time_s"] - base_rel, unit="s")
    df[f"datetime ({LOCAL_TIMEZONE})"] = df["datetime (UTC)"].dt.tz_convert(ZoneInfo(LOCAL_TIMEZONE))

    out = pd.DataFrame({
        "source_filename": os.path.basename(fit_file),
        "x-coordinate": df["position_long"],
        "y-coordinate": df["position_lat"],
        "datetime (UTC)": df["datetime (UTC)"],
        f"datetime ({LOCAL_TIMEZONE})": df[f"datetime ({LOCAL_TIMEZONE})"],
        "heading_fit": df["heading_fit"],
        "enhanced_altitude": df["enhanced_altitude"],
        "enhanced_speed": df["enhanced_speed"],
        "velocity": df.get("velocity", np.nan),
    })

    out = out.dropna(subset=["datetime (UTC)", "x-coordinate", "y-coordinate"])
    out = out.sort_values("datetime (UTC)").reset_index(drop=True)

    return out


# ========== FUNCTIONS FOR CALCULATION ==========

def calculate_heading(lat1, lon1, lat2, lon2):
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    fwd_azimuth = (fwd_azimuth + 360) % 360
    return fwd_azimuth


def heading(df):
    headings = []

    for i in range(len(df) - 1):
        lat1, lon1 = df.iloc[i]["y-coordinate"], df.iloc[i]["x-coordinate"]
        lat2, lon2 = df.iloc[i + 1]["y-coordinate"], df.iloc[i + 1]["x-coordinate"]

        if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2):
            head = calculate_heading(lat1, lon1, lat2, lon2)
        else:
            head = np.nan

        headings.append(head)

    headings.append(np.nan)
    df["heading"] = pd.Series(headings, index=df.index)
    return df


def solar_positions(df):
    df["instantaneous solar azimuth angle"] = np.nan
    df["instantaneous solar zenith angle"] = np.nan
    df["relative azimuth angle"] = np.nan

    valid_mask = (
        df["datetime (UTC)"].notna()
        & df["y-coordinate"].notna()
        & df["x-coordinate"].notna()
    )

    if valid_mask.any():
        solar_position = pvlib.solarposition.get_solarposition(
            df.loc[valid_mask, "datetime (UTC)"],
            df.loc[valid_mask, "y-coordinate"],
            df.loc[valid_mask, "x-coordinate"]
        )

        df.loc[valid_mask, "instantaneous solar azimuth angle"] = solar_position["azimuth"].values
        df.loc[valid_mask, "instantaneous solar zenith angle"] = solar_position["zenith"].values
        df.loc[valid_mask, "relative azimuth angle"] = (
            df.loc[valid_mask, "instantaneous solar azimuth angle"]
            - df.loc[valid_mask, "heading"]
        ) % 360

    return df


def cruise_speed(df):
    df["datetime (UTC)"] = pd.to_datetime(df["datetime (UTC)"], errors="coerce", utc=True)
    df["time_diff"] = df["datetime (UTC)"].diff().dt.total_seconds().fillna(0)

    distances = []
    speeds = []
    geod = Geodesic.WGS84

    for i in range(1, len(df)):
        lat1 = df.iloc[i - 1]["y-coordinate"]
        lon1 = df.iloc[i - 1]["x-coordinate"]
        lat2 = df.iloc[i]["y-coordinate"]
        lon2 = df.iloc[i]["x-coordinate"]

        if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2):
            g = geod.Inverse(lat1, lon1, lat2, lon2)
            distance = g["s12"]
        else:
            distance = np.nan

        time_diff = df.iloc[i]["time_diff"]
        if pd.notna(distance) and time_diff > 0:
            speed = distance / time_diff
        else:
            speed = 0

        distances.append(distance)
        speeds.append(speed)

    df["distance"] = [0] + distances
    df["speed"] = [0] + speeds
    df = df.drop(["time_diff", "distance"], axis=1)

    return df


# ========== INPUT ROUTER ==========

def extract_input_data(input_type, image_dir=None, gpx_file=None, fit_file=None):
    input_type = input_type.lower()

    if input_type == "virb":
        if not image_dir:
            raise ValueError("For INPUT_TYPE='virb', IMAGE_DIR must be set.")
        df = extract_from_virb(image_dir)

    elif input_type == "gpx":
        if not gpx_file:
            raise ValueError("For INPUT_TYPE='gpx', GPX_FILE must be set.")
        df = extract_from_gpx(gpx_file)

    elif input_type == "fit":
        if not fit_file:
            raise ValueError("For INPUT_TYPE='fit', FIT_FILE must be set.")
        df = extract_from_fit(fit_file)

    else:
        raise ValueError("INPUT_TYPE must be one of: 'virb', 'gpx', 'fit'.")

    return df


# ========== MAIN SCRIPT ==========

if __name__ == "__main__":
    df = extract_input_data(
        input_type=INPUT_TYPE,
        image_dir=IMAGE_DIR,
        gpx_file=GPX_FILE,
        fit_file=FIT_FILE
    )

    if df.empty:
        raise ValueError("No valid input data found.")

    # Ensure coordinate data is numeric
    df["x-coordinate"] = pd.to_numeric(df["x-coordinate"], errors="coerce")
    df["y-coordinate"] = pd.to_numeric(df["y-coordinate"], errors="coerce")

    # Ensure datetimes are parsed and sorted
    df["datetime (UTC)"] = pd.to_datetime(df["datetime (UTC)"], errors="coerce", utc=True)
    df = df.sort_values("datetime (UTC)").reset_index(drop=True)

    # Calculate parameters
    df = heading(df)
    df = solar_positions(df)
    df = cruise_speed(df)

    # Reorder columns
    local_dt_col = f"datetime ({LOCAL_TIMEZONE})"
    ordered_cols = [
        "source_filename",
        local_dt_col,
        "datetime (UTC)",
        "x-coordinate",
        "y-coordinate",
        "heading",
        "instantaneous solar azimuth angle",
        "instantaneous solar zenith angle",
        "relative azimuth angle",
        "speed",
    ]

    # Keep optional FIT columns if they exist
    optional_cols = ["heading_fit", "enhanced_altitude", "enhanced_speed", "velocity"]
    ordered_cols_extended = ordered_cols + [col for col in optional_cols if col in df.columns]

    df = df[ordered_cols_extended]

    # Make sure output directory exists
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved CSV to: {OUTPUT_CSV}")

    # Small sanity check
    print(df.head())
    print("Min UTC:", df["datetime (UTC)"].min())
    print("Max UTC:", df["datetime (UTC)"].max())
