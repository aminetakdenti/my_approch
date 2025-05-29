import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math


output_path = "/kaggle/working/dist"
attacks_obj = {}


def handle_large_files(file_path, chunk_size=100_000):
    total_rows = 0
    total_dropped = 0
    benign_count = 0
    attack_count = 0

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        before = chunk.shape[0]
        chunk = chunk.dropna()
        after = chunk.shape[0]

        total_rows += before
        total_dropped += before - after

        benign_count += chunk[chunk["Label"] == "BENIGN"].shape[0]
        attack_count += chunk[chunk["Label"] != "BENIGN"].shape[0]
    attack_obj[file_path] = {attack_count, benign_count, total_rows, total_dropped}


for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print("reading file", filename)

        file_path = os.path.join(dirname, filename)
        file_stats = os.stat(file_path)

        print(f"File Size in Bytes is {file_stats.st_size}")
        print(f"File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}")
        if math.floor(file_stats.st_size / (1024 * 1024 * 1024) / 4) > 0:
            handle_large_files(file_path)
        print("end")
        print(attack_obj)


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
