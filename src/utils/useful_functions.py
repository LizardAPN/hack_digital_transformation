import sys
from pathlib import Path

# Добавляем путь к утилитам для корректной работы импортов
utils_path = Path(__file__).resolve().parent.parent / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

# Настраиваем пути проекта
try:
    from path_resolver import setup_project_paths
    setup_project_paths()
except ImportError:
    # Если path_resolver недоступен, добавляем необходимые пути вручную
    src_path = Path(__file__).resolve().parent.parent
    paths_to_add = [src_path, src_path / "utils", src_path / "geo"]
    for path in paths_to_add:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
from pathlib import Path
from typing import Tuple, Optional, List, Any
import re
import shutil
from itertools import zip_longest

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def move_and_remove_files(source_dir: Path, destination_dir: Path, remove_after_move: bool = False) -> None:
    """
    Перемещает файлы из source_dir в destination_dir и удаляет source_dir если remove_after_move=True
    
    Args:
        source_dir: Путь к исходной директории
        destination_dir: Путь к целевой директории
        remove_after_move: Флаг удаления исходной директории после перемещения
    """
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


def extract_coordinates(coord_string: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Извлекает координаты из строки
    
    Args:
        coord_string: Строка с координатами в формате "coordinates=[lat, lon]"
        
    Returns:
        Кортеж с широтой и долготой или (None, None) если не найдены
    """
    pattern = r"coordinates=\[([\d.-]+),\s*([\d.-]+)\]"
    match = re.search(pattern, str(coord_string))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def merge_tables_with_tolerance(
    target: pd.DataFrame,
    real_data: pd.DataFrame,
    target_lat_name: str = "latitude",
    target_lot_name: str = "longitude",
    real_data_lat_name: str = "latitude",
    real_data_lot_name: str = "longitude",
    max_distance_meters: float = 100,
) -> pd.DataFrame:
    """
    Объединяет две таблицы по координатам с допустимым отклонением
    
    Args:
        target: Целевая таблица с координатами
        real_data: Таблица с реальными данными
        target_lat_name: Название колонки широты в target
        target_lot_name: Название колонки долготы в target
        real_data_lat_name: Название колонки широты в real_data
        real_data_lot_name: Название колонки долготы в real_data
        max_distance_meters: Максимальное расстояние в метрах для объединения
        
    Returns:
        Объединенный DataFrame с результатами
    """
    # Проверка существования колонок
    if target_lat_name not in target.columns:
        raise ValueError(f"Колонка {target_lat_name} не найдена в target")
    if target_lot_name not in target.columns:
        raise ValueError(f"Колонка {target_lot_name} не найдена в target")
    if real_data_lat_name not in real_data.columns:
        raise ValueError(f"Колонка {real_data_lat_name} не найдена в real_data")
    if real_data_lot_name not in real_data.columns:
        raise ValueError(f"Колонка {real_data_lot_name} не найдена в real_data")

    # Переименование колонок
    df1 = target.rename(
        columns={target.columns[0]: "filename", target_lat_name: "lat_target", target_lot_name: "lon_target"}
    )
    df2 = real_data.rename(
        columns={real_data.columns[0]: "camera_id", real_data_lat_name: "lat_real", real_data_lot_name: "lon_real"}
    )

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


def levenshtein_distance(string1: str, string2: str) -> int:
    """
    Вычисляет расстояние Левенштейна между двумя строками
    
    Args:
        string1: Первая строка
        string2: Вторая строка
        
    Returns:
        Расстояние Левенштейна между строками
        
    Examples:
        >>> levenshtein_distance('AATZ', 'AAAZ')
        1
        >>> levenshtein_distance('AATZZZ', 'AAAZ')
        3
    """
    distance = 0
    if len(string1) < len(string2):
        string1, string2 = string2, string1

    # Заменяем itertools.izip_longest на zip_longest для Python 3
    for i, v in zip_longest(string1, string2, fillvalue="-"):
        if i != v:
            distance += 1
    return distance
