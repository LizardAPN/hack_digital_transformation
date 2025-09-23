import os
import re
import shutil
import zipfile
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


def move_and_remove_files(source_dir, destination_dir, remove_after_move=False):
    if source_dir.exists() and source_dir.is_dir():
        for item in source_dir.iterdir():
            try:
                shutil.move(str(item), str(destination_dir))
                print(f"Перенесен: {item.name}")
            except Exception as e:
                print(f"Ошибка при переносе {item.name}: {e}")

        if remove_after_move:
            # Удаляем теперь пустую папку
            try:
                source_dir.rmdir()  # Удаляем пустую папку
                print(f"Пустая папка удалена: {source_dir}")
            except OSError:
                print(f"Папка не пуста, используем rmtree")
                shutil.rmtree(source_dir)


def extract_coordinates(coord_string):
    """Извлекает координаты из строки"""
    pattern = r"coordinates=\[([\d.-]+),\s*([\d.-]+)\]"
    match = re.search(pattern, str(coord_string))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def merge_tables_with_tolerance(df1, df2, max_distance_meters=100):

    df1 = df1.rename(columns={df1.columns[0]: "filename", df1.columns[1]: "lat1", df1.columns[2]: "lon1"})

    df2 = df2.rename(columns={df2.columns[0]: "camera_id", df2.columns[1]: "lat2", df2.columns[2]: "lon2"})

    df1["key"] = 1
    df2["key"] = 1
    combinations = pd.merge(df1, df2, on="key").drop("key", axis=1)

    combinations["distance_m"] = combinations.apply(
        lambda row: calculate_distance(row["lat1"], row["lon1"], row["lat2"], row["lon2"]), axis=1
    )

    closest_matches = combinations.loc[combinations.groupby("filename")["distance_m"].idxmin()]

    result = closest_matches[closest_matches["distance_m"] <= max_distance_meters].copy()

    result = result.sort_values("distance_m")

    result = result.reset_index(drop=True)

    return result
