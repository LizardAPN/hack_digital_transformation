 import json
import os
import pickle

import faiss

from utils.config import DATA_PATHS, s3_manager


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
                mapping = pickle.load(f)
            print(f"✓ Маппинг: {len(mapping)} записей")
        except Exception as e:
            print(f"✗ Ошибка загрузки маппинга: {e}")
    else:
        print("✗ Файл маппинга не найден")

    # Проверка наличия и целостности метаданных сборки
    if os.path.exists(DATA_PATHS["metadata_file"]):
        try:
            with open(DATA_PATHS["metadata_file"], "r") as f:
                metadata = json.load(f)
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
