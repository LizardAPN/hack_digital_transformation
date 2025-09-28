import json
import os
import pickle

import faiss

from utils.config import DATA_PATHS, s3_manager


def monitor_database():
    """Мониторинг состояния базы данных"""
    print("=== МОНИТОРИНГ БАЗЫ ДАННЫХ ===")

    # Проверка FAISS индекса
    if os.path.exists(DATA_PATHS["faiss_index"]):
        try:
            index = faiss.read_index(DATA_PATHS["faiss_index"])
            print(f"✓ FAISS индекс: {index.ntotal} изображений")
        except Exception as e:
            print(f"✗ Ошибка загрузки FAISS индекса: {e}")
    else:
        print("✗ FAISS индекс не найден")

    # Проверка маппинга
    if os.path.exists(DATA_PATHS["mapping_file"]):
        try:
            with open(DATA_PATHS["mapping_file"], "rb") as f:
                mapping = pickle.load(f)
            print(f"✓ Маппинг: {len(mapping)} записей")
        except Exception as e:
            print(f"✗ Ошибка загрузки маппинга: {e}")
    else:
        print("✗ Файл маппинга не найден")

    # Проверка метаданных
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

    # Проверка S3
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
    monitor_database()
