import os
import re
import shutil
import zipfile
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


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


def merge_tables_with_tolerance(df1, df2, max_distance_meters=100):
    # Переименование колонок
    df1 = df1.rename(columns={df1.columns[0]: "filename", df1.columns[2]: "lat_target", df1.columns[1]: "lon_target"})
    df2 = df2.rename(columns={df2.columns[0]: "camera_id", df2.columns[1]: "lat_real", df2.columns[2]: "lon_real"})

    # Преобразование координат в радианы для сферического расстояния
    coords1 = np.radians(df1[["lat_target", "lon_target"]].values)
    coords2 = np.radians(df2[["lat_real", "lon_real"]].values)

    # Создание KDTree для быстрого поиска
    tree = KDTree(coords2)

    # Поиск ближайшей точки в df2 для каждой точки в df1
    distances, indices = tree.query(coords1, k=1)  # k=1 для одной ближайшей точки

    # Преобразование расстояний из радиан в метры (Earth's radius ≈ 6371000 m)
    distances_m = distances * 6371000

    # Создание результирующего DataFrame
    result = df1.copy()
    result["distance_m"] = distances_m
    result["camera_id"] = df2.iloc[indices]["camera_id"].values
    result["lat_real"] = df2.iloc[indices]["lat_real"].values
    result["lon_real"] = df2.iloc[indices]["lon_real"].values

    # Фильтрация по максимальному расстоянию
    result = result[result["distance_m"] <= max_distance_meters].sort_values("distance_m")

    return result.reset_index(drop=True)
