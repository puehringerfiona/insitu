'''
extracts metadata (filename, datetime, inclination v, x and y) from ramses files
'''

import pandas as pd
import glob
import os
import pytz
from pathlib import Path

# Define the folder containing RAMSES .dat files
input_folder = Path(r"C:\Users\puehrifi\Documents\campaigns\20240619_WAL\insitu_data\raw_data\ramses\Raw")
output_file  = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\metadata\20240619_WAL\20240619_WAL_ramses_metadata.csv")

# Files
file_list_ramses = sorted(input_folder.glob("*.dat"))

inclination_records = []

def extract_inclination_data(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    metadata = {}
    data_start = None

    for i, line in enumerate(content):
        line = line.strip()

        # Detect Inclination block
        if line.startswith("IDDataType") and "Inclination" in line:
            metadata = {"filename": os.path.basename(file_path)}

        elif line.startswith("DateTime") and metadata:
            metadata["DateTime"] = line.split('=', 1)[1].strip()

        elif line.startswith("[DATA]") and metadata:
            data_start = i + 1

        elif line.startswith("[END] of [DATA]") and metadata and data_start:
            try:
                incl_values = content[data_start].strip().split()
                if len(incl_values) >= 3:
                    metadata["Incl_X"] = float(incl_values[0])
                    metadata["Incl_Y"] = float(incl_values[1])
                    metadata["Incl_V"] = float(incl_values[2])
                    inclination_records.append(metadata.copy())
            except (IndexError, ValueError):
                pass

            # Reset for next block
            metadata = {}
            data_start = None



# Iterate through each RAMSES file and extract inclination data
for file_path in file_list_ramses:
    extract_inclination_data(file_path)

# Convert the list of dictionaries into a Pandas DataFrame
df_inclination = pd.DataFrame(inclination_records)

# Ensure DateTime is CET/CEST and sort earliest to latest
cet = pytz.timezone("Europe/Berlin")

df_inclination["DateTime"] = pd.to_datetime(df_inclination["DateTime"], errors="coerce")
df_inclination["DateTime"] = df_inclination["DateTime"].dt.tz_localize(cet, ambiguous="NaT", nonexistent="NaT")

df_inclination = df_inclination.dropna(subset=["DateTime"])
df_inclination = df_inclination.sort_values(by="DateTime").drop_duplicates()

# Save to CSV
df_inclination.to_csv(output_file, index=False)
