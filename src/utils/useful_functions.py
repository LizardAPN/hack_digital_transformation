import collections
import itertools
import os
import re
import shutil
import zipfile
from itertools import zip_longest
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def move_and_remove_files(source_dir, destination_dir, remove_after_move=False):
    """
    Перемещает файлы из исходной директории в целевую и при необходимости удаляет исходную директорию.

    Функция перемещает все элементы из указанной исходной директории в целевую директорию.
    Если параметр remove_after_move установлен в True, то после перемещения исходная 
    директория удаляется.

    Параметры
    ----------
    source_dir : Path
        Путь к исходной директории.
    destination_dir : Path
        Путь к целевой директории.
    remove_after_move : bool, optional
        Флаг, указывающий на необходимость удаления исходной директории после перемещения 
        (по умолчанию False).

    Примеры
    --------
    >>> from pathlib import Path
    >>> move_and_remove_files(Path('/path/to/source'), Path('/path/to/destination'))
    Перенесен: file1.txt
    Перенесен: file2.txt
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


def extract_coordinates(coord_string):
    """
    Извлекает координаты из строки.

    Функция извлекает широту и долготу из строки формата "coordinates=[широта, долгота]".

    Параметры
    ----------
    coord_string : str
        Строка, содержащая координаты в формате "coordinates=[число, число]".

    Возвращает
    -------
    tuple
        Кортеж из двух float значений (широта, долгота) или (None, None), если 
        координаты не найдены.

    Примеры
    --------
    >>> extract_coordinates("coordinates=[55.7558, 37.6173]")
    (55.7558, 37.6173)
    >>> extract_coordinates("no coordinates here")
    (None, None)
    """
    pattern = r"coordinates=\[([\d.-]+),\s*([\d.-]+)\]"
    match = re.search(pattern, str(coord_string))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def merge_tables_with_tolerance(
    target,
    real_data,
    target_lat_name="latitude",
    target_lot_name="longitude",
    real_data_lat_name="latitude",
    real_data_lot_name="longitude",
    max_distance_meters=100,
):
    """
    Объединяет две таблицы по координатам с учетом допуска на расстояние.

    Функция объединяет две таблицы данных, сопоставляя записи по географическим координатам.
    Для каждой записи в таблице target находится ближайшая запись в таблице real_data 
    в пределах заданного максимального расстояния.

    Параметры
    ----------
    target : pandas.DataFrame
        Таблица с данными, к которым будут присоединены данные из real_data.
    real_data : pandas.DataFrame
        Таблица с реальными данными, которые будут присоединены к target.
    target_lat_name : str, optional
        Название столбца с широтой в таблице target (по умолчанию "latitude").
    target_lot_name : str, optional
        Название столбца с долготой в таблице target (по умолчанию "longitude").
    real_data_lat_name : str, optional
        Название столбца с широтой в таблице real_data (по умолчанию "latitude").
    real_data_lot_name : str, optional
        Название столбца с долготой в таблице real_data (по умолчанию "longitude").
    max_distance_meters : int, optional
        Максимальное расстояние в метрах для сопоставления записей (по умолчанию 100).

    Возвращает
    -------
    pandas.DataFrame
        Результирующая таблица с объединенными данными, отсортированная по расстоянию.

    Исключения
    ----------
    ValueError
        Возникает, если указанные столбцы с координатами не найдены в соответствующих таблицах.
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


def levenshtein_distance(string1, string2):
    """
    Вычисляет расстояние Левенштейна между двумя строками.

    Расстояние Левенштейна — это минимальное количество односимвольных 
    операций (вставки, удаления или замены), необходимых для преобразования 
    одной строки в другую.

    Параметры
    ----------
    string1 : str
        Первая строка для сравнения.
    string2 : str
        Вторая строка для сравнения.

    Возвращает
    -------
    int
        Расстояние Левенштейна между строками.

    Примеры
    --------
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
