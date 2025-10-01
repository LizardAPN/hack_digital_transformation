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
import json
import os
import pickle
from typing import Any, Dict, List, Union

import faiss

from src.utils.config import DATA_PATHS, s3_manager


def monitor_database():
    """
    Выполняет диагностику состояния базы данных системы поиска изображений.

    Функция проверяет доступность и целостность ключевых компонентов системы:
    - FAISS индекса для поиска похожих изображений
    - Файла сопоставления идентификаторов изображений
    - Метаданных сборки системы
    - Подключения к облачному хранилищу S3

    Параметры
    ----------
    Отсутствуют

    Вывод
    -------
    None
        Выводит информацию о состоянии системы в стандартный поток вывода.

    Примеры
    --------
    >>> monitor_database()
    === МОНИТОРИНГ БАЗЫ ДАННЫХ ===
    ✓ FAISS индекс: 10000 изображений
    ✓ Маппинг: 10000 записей
    ...
    """
    print("=== МОНИТОРИНГ БАЗЫ ДАННЫХ ===")

    # Проверка наличия и целостности FAISS индекса
    if os.path.exists(DATA_PATHS["faiss_index"]):
        try:
            index = faiss.read_index(DATA_PATHS["faiss_index"])
            print(f"✓ FAISS индекс: {index.ntotal} изображений")
        except Exception as e:
            print(f"✗ Ошибка загрузки FAISS индекса: {e}")
    else:
        print("✗ FAISS индекс не найден")

    # Проверка наличия и целостности файла сопоставления идентификаторов
    if os.path.exists(DATA_PATHS["mapping_file"]):
        try:
            with open(DATA_PATHS["mapping_file"], "rb") as f:
                mapping: Dict[Any, Any] = pickle.load(f)
            print(f"✓ Маппинг: {len(mapping)} записей")
        except Exception as e:
            print(f"✗ Ошибка загрузки маппинга: {e}")
    else:
        print("✗ Файл маппинга не найден")

    # Проверка наличия и целостности метаданных сборки
    if os.path.exists(DATA_PATHS["metadata_file"]):
        try:
            with open(DATA_PATHS["metadata_file"], "r") as f:
                metadata: Dict[str, Union[str, int, float, bool, List[Any], Dict[Any, Any]]] = json.load(f)
            print("✓ Метаданные сборки:")
            for key, value in metadata.items():
                if key != "processing_stats":
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"✗ Ошибка загрузки метаданных: {e}")
    else:
        print("✗ Метаданные не найдены")

    # Проверка подключения и доступности облачного хранилища S3
    try:
        total_s3_files = len(s3_manager.list_files(prefix=""))
        print(f"✓ S3 бакет: {total_s3_files} файлов")

        upload_stats = s3_manager.get_upload_stats()
        download_stats = s3_manager.get_download_stats()
        print(f"✓ Статистика S3 загрузки: {upload_stats}")
        print(f"✓ Статистика S3 скачивания: {download_stats}")

    except Exception as e:
        print(f"✗ Ошибка подключения к S3: {e}")


if __name__ == "__main__":
    # Точка входа для прямого запуска скрипта диагностики
    monitor_database()
