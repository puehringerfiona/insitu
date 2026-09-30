import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from pysolar.solar import get_altitude
import rrs_estimation_3C_model as m3c
import os
import matplotlib.pyplot as plt
import re

# ------------------ Parameters ------------------

low_lim = 350  # nm
up_lim = 900   # nm
wl_res = 1     # nm

input_folder = r'C:\Users\puehrifi\Documents\insitu\20231008_ZRH\preprocessing\20231008_ZRH_ramses_spectra'
output_path = r'C:\Users\puehrifi\Documents\insitu\20231008_ZRH\preprocessing\3C_output'
meta_file = r'C:\Users\puehrifi\Documents\insitu\20231008_ZRH\final_metadata\20231008_ZRH_metadata_with_Rrs_L9.csv'
df_meta = pd.read_csv(meta_file, sep=None, engine='python')
df_meta.columns = df_meta.columns.str.strip()
df_meta.columns = df_meta.columns.str.replace('\ufeff', '', regex=False)

print("Columns found:")
print([repr(c) for c in df_meta.columns])

if 'timestamp (UTC)' not in df_meta.columns:
    raise KeyError(f"'timestamp (UTC)' not found. Available columns: {df_meta.columns.tolist()}")

df_meta = df_meta.sort_values('timestamp (UTC)').reset_index(drop=True)
os.makedirs(output_path, exist_ok=True)

def find_spectral_file(input_folder, labels, file_description):
    candidates = []
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".csv"):
            continue
        stem = os.path.splitext(filename)[0]
        for label in labels:
            if re.search(rf"(^|[_\-\s]){re.escape(label)}($|[_\-\s])", stem, flags=re.IGNORECASE):
                candidates.append(os.path.join(input_folder, filename))
                break

    if not candidates:
        raise FileNotFoundError(
            f"No {file_description} CSV found in {input_folder}. "
            f"Expected filename token: {', '.join(labels)}."
        )
    if len(candidates) > 1:
        formatted = "\n  ".join(candidates)
        raise ValueError(
            f"Found multiple {file_description} CSV files in {input_folder}; "
            f"please keep only one matching file or narrow input_folder:\n  {formatted}"
        )
    return candidates[0]


Ed_fname = find_spectral_file(input_folder, ["Ed"], "Ed")
Lu_fname = find_spectral_file(input_folder, ["Lu", "Lw"], "Lu/Lw")
Ls_fname = find_spectral_file(input_folder, ["Ls", "Ld"], "Ls/Ld")

print(f"Using Ed file: {Ed_fname}")
print(f"Using Lu/Lw file: {Lu_fname}")
print(f"Using Ls/Ld file: {Ls_fname}")

#latitude = 47.2475
#longitude = 8.6775
europ_zurich = pytz.timezone("Europe/Zurich")
#time = datetime(2023, 10, 8, 11, 00, 00, 0, tzinfo=europ_zurich)

# ------------------ Functions ------------------

def wl_interpolator_v02(df_in, low_lim, up_lim, wl_res):
    wl_interp = np.arange(low_lim, up_lim + wl_res, wl_res)
    df_interp = df_in.reindex(df_in.index.union(wl_interp))
    df_interp = df_interp.interpolate('index', limit_direction='both').loc[wl_interp]
    return df_interp

# def csv_wl_read_interp(fname, low_lim, up_lim, wl_res):
#     df_raw = pd.read_csv(fname, index_col=0)
#     id_low_lim = np.min(np.where(df_raw.index > low_lim))
#     id_up_lim = np.max(np.where(df_raw.index < up_lim))
#     df_raw = df_raw.iloc[id_low_lim - 1:id_up_lim + 2]
#     df_interp = wl_interpolator_v02(df_raw, low_lim, up_lim, wl_res)
#     df_interp = df_interp.interpolate('linear', limit_direction='both', axis=1)
#     df_interp['median'] = df_interp.quantile(q=0.5, axis=1)
#     df_interp['q25'] = df_interp.quantile(q=0.25, axis=1)
#     df_interp['q75'] = df_interp.quantile(q=0.75, axis=1)
#     return df_interp

# had to remove first timestamp column and nan rows for ZRH20231008, first two for WAL

def csv_wl_read_interp(fname, low_lim, up_lim, wl_res):
    df_raw = pd.read_csv(fname, index_col=0)

    # Make wavelength index numeric
    df_raw.index = pd.to_numeric(df_raw.index, errors='coerce')

    # Drop rows where wavelength could not be parsed
    df_raw = df_raw[~df_raw.index.isna()]

    # Sort wavelengths
    df_raw = df_raw.sort_index()

    # Try to parse and sort timestamp-like columns
    col_times = pd.to_datetime(df_raw.columns, errors='coerce')
    if col_times.notna().any():
        sorted_pairs = sorted(
            [(t, c) for t, c in zip(col_times, df_raw.columns) if pd.notna(t)],
            key=lambda x: x[0]
        )
        sorted_cols = [c for _, c in sorted_pairs]

        # Keep non-datetime columns too, if any
        nondatetime_cols = [c for c, t in zip(df_raw.columns, col_times) if pd.isna(t)]
        df_raw = df_raw[nondatetime_cols + sorted_cols]

    # Remove most recent timestamps if needed
    # if "ZRH" in fname:
    #     print("Detected 'ZRH' in filename — removing most recent timestamp column")
    #     if len(df_raw.columns) >= 1:
    #         df_raw = df_raw.iloc[:, 1:]
    # elif "WAL" in fname:
    #     print("Detected 'WAL' in filename — removing two most recent timestamp columns")
    #     if len(df_raw.columns) >= 2:
    #         df_raw = df_raw.iloc[:, 2:]

    # Limit wavelength range safely
    wl_mask = (df_raw.index >= low_lim) & (df_raw.index <= up_lim)
    df_raw = df_raw.loc[wl_mask]

    # Interpolate along wavelength
    df_interp = wl_interpolator_v02(df_raw, low_lim, up_lim, wl_res)

    # Interpolate across timestamps
    df_interp = df_interp.interpolate('linear', limit_direction='both', axis=1)

    # Compute quantiles
    df_interp['median'] = df_interp.quantile(q=0.5, axis=1)
    df_interp['q25'] = df_interp.quantile(q=0.25, axis=1)
    df_interp['q75'] = df_interp.quantile(q=0.75, axis=1)

    return df_interp


# ------------------ Load & Interpolate Data ------------------

data_Ed = csv_wl_read_interp(Ed_fname, low_lim, up_lim, wl_res)
data_Lu = csv_wl_read_interp(Lu_fname, low_lim, up_lim, wl_res)
data_Ls = csv_wl_read_interp(Ls_fname, low_lim, up_lim, wl_res)

for name, df in zip(["Ed", "Lw", "Lsky"], [data_Ed, data_Lu, data_Ls]):
    print(f"{name} has NaNs? {df.isna().any().any()}")
    print(f"{name} NaN rows: {df[df.isna().any(axis=1)].index.tolist()}")
    print(f"{name} NaN columns: {df.columns[df.isna().any()].tolist()}")

threshold = 1e-6  # adjust as needed
for name, df in zip(["Ed", "Lw", "Lsky"], [data_Ed, data_Lu, data_Ls]):
    zero_mask = (df < threshold).any(axis=1)
    if zero_mask.any():
        print(f"{name} has very low values at wavelengths: {df.index[zero_mask].tolist()}")

print("Ed wavelengths:", data_Ed.index.min(), "-", data_Ed.index.max(), "count:", len(data_Ed))
print("Lu wavelengths:", data_Lu.index.min(), "-", data_Lu.index.max(), "count:", len(data_Lu))
print("Ls wavelengths:", data_Ls.index.min(), "-", data_Ls.index.max(), "count:", len(data_Ls))

common_wl = data_Ed.index.intersection(data_Lu.index).intersection(data_Ls.index)
print("Common wavelengths:", len(common_wl))

print("data_Ed columns:", len(data_Ed.columns))
print("data_Lu columns:", len(data_Lu.columns))
print("data_Ls columns:", len(data_Ls.columns))

# Check sample times vs metadata times
sample_times = pd.to_datetime(data_Ed.columns, errors='coerce')
nan_times = sample_times[pd.isna(sample_times)]
print(f"Number of invalid timestamps in Ed: {len(nan_times)}")

# ------------------ Plot Ed, Lu, Ls ---------------------------

fig_size = (8,10)
xy_lbl = ['$\lambda \ [nm]$','$\mathrm{Irradiance \ [mW m^{-2} nm^{-1}]}$','$\mathrm{Radiance \ [mW m^{-2} nm^{-1} Sr^{-1}]}$'] # [xlaebl, ylabel]
xtick_rotation = 0
font_feature = ['Helvetica', 16, 14]
x_lim = [350,800]#[x[0],x[-1]]
y_lim1 = [200,1200]
y_lim2 = [0,80]

xtick_dateformat = None # None or e.g., '%d-%m %H'

plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)

if xtick_rotation != 0:
    plt.xticks(rotation=xtick_rotation, ha=u"right",fontname=font_feature[0],fontsize=font_feature[2])
else:
    plt.xticks(fontname=font_feature[0],fontsize=font_feature[2])
plt.yticks(fontname=font_feature[0],fontsize=font_feature[2])

ax1.plot(data_Ed.index,data_Ed['median'], color = 'k',lw=2, label='$E_d$')
ax1.fill_between(data_Ed.index, data_Ed['q25'], data_Ed['q75'],alpha=0.3, facecolor='k')

leg = plt.legend(loc = 'best',facecolor = 'k',frameon=False, prop={"size":font_feature[1]-4,'family':font_feature[0]})
plt.setp(leg.get_texts(), color='k')

ax1.set_xlim(xmin=x_lim[0], xmax=x_lim[1])
ax1.set_ylim(ymin=y_lim1[0], ymax=y_lim1[1])

# plt.title('Point 1 - 27.08.2020',fontname=font_feature[0],fontsize=font_feature[2])
# ax1.set_xlabel(xy_lbl[0], fontsize= font_feature[1],fontname=font_feature[0])
ax1.set_ylabel(xy_lbl[1], fontsize=font_feature[1],fontname=font_feature[0])

ax2 = fig.add_subplot(212)

if xtick_rotation != 0:
    plt.xticks(rotation=xtick_rotation, ha=u"right", fontname=font_feature[0], fontsize=font_feature[2])
else:
    plt.xticks(fontname=font_feature[0], fontsize=font_feature[2])
plt.yticks(fontname=font_feature[0], fontsize=font_feature[2])

ax2.plot(data_Ls.index, data_Ls['median'], color='r', lw=2, label=r'$L_{sky}$')
ax2.plot(data_Lu.index, data_Lu['median'], color='b', lw=2, label=r'$L_w$')

ax2.fill_between(data_Lu.index, data_Lu['q25'], data_Lu['q75'], alpha=0.3, facecolor='b')
ax2.fill_between(data_Ls.index, data_Ls['q25'], data_Ls['q75'], alpha=0.3, facecolor='r')

leg = plt.legend(
    loc='best',
    facecolor='k',
    frameon=False,
    prop={"size": font_feature[1]-4, 'family': font_feature[0]}
)
plt.setp(leg.get_texts(), color='k')

ax2.set_xlim(xmin=x_lim[0], xmax=x_lim[1])
ax2.set_ylim(ymin=y_lim2[0], ymax=y_lim2[1])

ax2.set_xlabel(xy_lbl[0], fontsize=font_feature[1], fontname=font_feature[0])
ax2.set_ylabel(xy_lbl[2], fontsize=font_feature[1], fontname=font_feature[0])

plt.show()

# ------------------ Compute Sun Zenith Angle ------------------

#sunzenith = 90.0 - get_altitude(latitude, longitude, time)

# ------------------ Compute Dynamic Sun Zenith Angles (Matched to Each Spectral Sample) ------------------

# --- Convert metadata timestamps to timezone-aware datetimes ---
if np.issubdtype(df_meta['timestamp (UTC)'].dtype, np.number):
    # Convert UNIX timestamps to timezone-aware datetimes
    df_meta['datetime'] = df_meta['timestamp (UTC)'].apply(
        lambda ts: datetime.fromtimestamp(ts, pytz.timezone('Europe/Zurich'))
    )
else:
    # If timestamps are already datetime strings, parse them
    df_meta['datetime'] = pd.to_datetime(df_meta['timestamp (UTC)'], errors='coerce').dt.tz_localize('Europe/Zurich')

df_meta['x-coordinate'] = pd.to_numeric(df_meta['x-coordinate'], errors='coerce')
df_meta['y-coordinate'] = pd.to_numeric(df_meta['y-coordinate'], errors='coerce')
df_meta = df_meta.dropna(subset=['datetime', 'x-coordinate', 'y-coordinate']).reset_index(drop=True)
if df_meta.empty:
    raise ValueError("No metadata rows with valid datetime and coordinates are available for sun zenith matching.")

# --- Compute solar zenith for every metadata entry ---
df_meta['sunzenith'] = [
    90.0 - get_altitude(
        df_meta.loc[ii, 'y-coordinate'],
        df_meta.loc[ii, 'x-coordinate'],
        df_meta.loc[ii, 'datetime']
    )
    for ii in range(len(df_meta))
]

print(f"Computed {len(df_meta)} dynamic sun zenith angles from metadata.")

# ------------------ Match Each Spectral Sample (data_Ed) to Nearest Metadata Timestamp ------------------

# Parse the spectral sample timestamps from column headers
# (e.g., '2025-03-03 12:31:36')
sample_times = pd.to_datetime(data_Ed.columns, format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Drop invalid timestamps
sample_times = sample_times[pd.notna(sample_times)]

# Localize to Europe/Zurich (same as metadata)
# Note: use tz_localize, not tz_convert — these timestamps are UTC-naive
sample_times = sample_times.tz_localize('Europe/Zurich')

print(f"Found {len(sample_times)} valid timestamps in data_Ed columns.")

# Compute sunzenith per sample by finding the nearest metadata timestamp
sunzenith_per_sample = []
for t in sample_times:
    idx = (df_meta['datetime'] - t).abs().idxmin()
    sunzenith_per_sample.append(df_meta.loc[idx, 'sunzenith'])
sunzenith = np.array(sunzenith_per_sample, dtype=float)

# --- Save for later use ---
np.savez(os.path.join(output_path, 'sunzenith_field.npz'), sza=sunzenith)
print(f"Matched {len(sunzenith)} sun zenith angles to {len(sample_times)} spectral samples.")

# Parse and keep only valid timestamp columns
timestamp_cols = [c for c in data_Ed.columns if pd.notna(pd.to_datetime(c, errors='coerce'))]

if len(timestamp_cols) != len(sunzenith):
    raise ValueError(
        f"Timestamp column count ({len(timestamp_cols)}) does not match sun zenith count ({len(sunzenith)})."
    )

finite_sunzenith = np.isfinite(sunzenith)
if not finite_sunzenith.all():
    dropped = np.count_nonzero(~finite_sunzenith)
    print(f"Dropping {dropped} spectral samples with non-finite sun zenith.")
    timestamp_cols = [col for col, keep in zip(timestamp_cols, finite_sunzenith) if keep]
    sunzenith = sunzenith[finite_sunzenith]

# Keep the same columns in all matrices
data_Ed = data_Ed[timestamp_cols]
data_Lu = data_Lu[timestamp_cols]
data_Ls = data_Ls[timestamp_cols]

print(f"Aligned sample count: {len(timestamp_cols)}")

# ------------------ Run 3C Model ------------------

Rrs_output_3C, Lu_Ed_obs_3C, Lu_Ed_model_3C = m3c.my_3C_v01(data_Ed, data_Lu, data_Ls, output_path, sunzenith)

# Add quantiles to results if not already present
for df in [Rrs_output_3C, Lu_Ed_obs_3C, Lu_Ed_model_3C]:
    cols = [c for c in df.columns if re.match(r'\d{4}-\d{2}-\d{2}', c)]
    df['median'] = df[cols].quantile(0.5, axis=1)
    df['q25'] = df[cols].quantile(0.25, axis=1)
    df['q75'] = df[cols].quantile(0.75, axis=1)

# ------------------ Save Outputs ------------------

Rrs_output_3C.to_csv(os.path.join(output_path, 'Rrs_output_3C.csv'), index_label='wavelength_nm')
Rrs_output_3C.T.to_csv(os.path.join(output_path, 'Rrs_output_3C_transposed_for_excel.csv'), index_label='sample')
Lu_Ed_obs_3C.to_csv(os.path.join(output_path, 'Lu_Ed_obs_3C.csv'), index_label='wavelength_nm')
Lu_Ed_model_3C.to_csv(os.path.join(output_path, 'Lu_Ed_model_3C.csv'), index_label='wavelength_nm')

# Convert columns to floats
for col in ['q25', 'q75', 'median']:
    Rrs_output_3C[col] = pd.to_numeric(Rrs_output_3C[col], errors='coerce')

# ------------------ Plot Rrs ------------------

fig_size = (8,10)
xy_lbl = ['$\lambda \ [nm]$','$\mathrm{R_{rs}}$'] # [xlaebl, ylabel]
xtick_rotation = 0
font_feature = ['Times New Roman', 22, 18] # [fontname, fontsize_labels, fontsize_ticks]
x_lim = [350,900]#[x[0],x[-1]]
y_lim = [0,0.014]

xtick_dateformat = None # None or e.g., '%d-%m %H'

plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=fig_size)
ax2 = fig.add_subplot(211)

if xtick_rotation != 0:
    plt.xticks(rotation=xtick_rotation, ha=u"right",fontname=font_feature[0],fontsize=font_feature[2])
else:
    plt.xticks(fontname=font_feature[0],fontsize=font_feature[2])
plt.yticks(fontname=font_feature[0],fontsize=font_feature[2])

# Clean DataFrame by dropping rows with non-finite values
valid_mask = np.isfinite(Rrs_output_3C['q25']) & np.isfinite(Rrs_output_3C['q75']) & np.isfinite(Rrs_output_3C['median'])
Rrs_clean = Rrs_output_3C[valid_mask]

ax2.plot(Rrs_clean.index, Rrs_clean['median'], lw=2, label='3C')
ax2.fill_between(Rrs_clean.index, Rrs_clean['q25'], Rrs_clean['q75'], alpha=0.3, facecolor='b')

leg = plt.legend(loc = 'best',facecolor = 'k',frameon=False, prop={"size":font_feature[1]-4,'family':font_feature[0]})
plt.setp(leg.get_texts(), color='k')

ax2.set_xlim(xmin=x_lim[0], xmax=x_lim[1])
ax2.set_ylim(ymin=y_lim[0], ymax=y_lim[1])

ax2.set_xlabel(xy_lbl[0], fontsize= font_feature[1],fontname=font_feature[0])
ax2.set_ylabel(xy_lbl[1], fontsize=font_feature[1],fontname=font_feature[0])

plt.show()
