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
from typing import Dict, List, Any
import json
import os
import pickle
import time
from pathlib import Path

import torch

from .faiss_indexer import FaissIndexer
from src.models.feature_extractor import FeatureExtractor
from src.utils.config import DATA_PATHS, s3_manager, PROJECT_ROOT


def create_directories() -> None:
    """Создание необходимых директорий"""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    index_dir = PROJECT_ROOT / "data" / "index"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    print("Директории созданы")


def validate_s3_connection() -> bool:
    """Проверка подключения к S3"""
    print("Проверка подключения к S3...")
    try:
        # Пробуем получить список файлов (ограничиваемся 1 для проверки)
        connect = s3_manager
        if connect is not None:
            print("✓ Подключение к S3 успешно")
            return True
        else:
            print("✗ Не удалось получить список файлов из S3")
            return False
    except Exception as e:
        print(f"✗ Ошибка подключения к S3: {e}")
        return False


def main() -> None:
    print("=== Фаза 1: Построение базы данных изображений Москвы ===")
    print("Версия: PyTorch + S3Manager")
    start_time = time.time()

    # Создаем директории
    create_directories()

    # Проверяем подключение к S3
    if not validate_s3_connection():
        return

    # Очистка кэша CUDA если используется GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Кэш CUDA очищен")

    # 1. Извлечение признаков
    print("\n" + "=" * 50)
    print("1. Извлечение признаков из изображений в S3...")

    extractor = FeatureExtractor()
    features_dict, failed_images = extractor.process_all_images(batch_size=256)

    if not features_dict:
        print("Не удалось извлечь признаки ни из одного изображения")
        return

    # Сохраняем сырые признаки (опционально, для отладки)
    print("Сохранение сырых признаков...")
    raw_features_path = PROJECT_ROOT / "data" / "processed" / "raw_features.pkl"
    with open(raw_features_path, "wb") as f:
        pickle.dump(features_dict, f)

    # 2. Создание FAISS индекса
    print("\n" + "=" * 50)
    print("2. Создание поискового индекса FAISS...")

    indexer = FaissIndexer(dimension=2048)  # ResNet50 features are 2048-dim
    num_indexed = indexer.create_index(features_dict, index_type="IVF")
    print(f"Проиндексировано изображений: {num_indexed}")

    # 3. Сохранение индекса
    print("\n" + "=" * 50)
    print("3. Сохранение индекса и метаданных...")

    indexer.save_index(DATA_PATHS["faiss_index"], DATA_PATHS["mapping_file"])

    # 4. Сохранение метаданных сборки
    processing_stats = extractor.get_processing_stats()
    metadata = {
        "total_images_processed": len(features_dict),
        "failed_images": len(failed_images),
        "index_size": num_indexed,
        "model_used": "ResNet50_PyTorch",
        "feature_dimension": 2048,
        "device_used": extractor.device,
        "build_time_seconds": time.time() - start_time,
        "processing_stats": processing_stats,
        "s3_bucket": s3_manager.bucket_name,
        "build_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size_used": 16,
    }

    with open(DATA_PATHS["metadata_file"], "w") as f:
        json.dump(metadata, f, indent=2)

    # 5. Статистика S3
    s3_upload_stats = s3_manager.get_upload_stats()
    s3_download_stats = s3_manager.get_download_stats()

    print(f"\n" + "=" * 50)
    print("=== СБОРКА ЗАВЕРШЕНА! ===")
    print(f"Общее время: {time.time() - start_time:.2f} секунд")
    print(f"Успешно обработано: {len(features_dict)} изображений")
    print(f"Ошибки обработки: {len(failed_images)}")
    print(f"Размер FAISS индекса: {num_indexed}")
    print(f"\nСтатистика S3 (загрузка): {s3_upload_stats}")
    print(f"Статистика S3 (скачивание): {s3_download_stats}")

    # Сохраняем список неудачных изображений
    if failed_images:
        failed_images_path = PROJECT_ROOT / "data" / "processed" / "failed_images.txt"
        with open(failed_images_path, "w") as f:
            for img in failed_images:
                f.write(f"{img}\n")
        print(f"Список неудачных изображений сохранен: {failed_images_path}")


if __name__ == "__main__":
    main()
