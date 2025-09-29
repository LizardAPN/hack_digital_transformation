import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch

current_file = Path(__file__).resolve()
project_root_in_cloud = Path("/job")  # Явно указываем корень в облаке
local_project_root = current_file.parent.parent

# Выбираем корень в зависимости от окружения
# Проверяем, находимся ли мы в среде DataSphere (существует ли папка /job)
if project_root_in_cloud.exists():
    ROOT_DIR = project_root_in_cloud
    print("✓ Обнаружена среда DataSphere. Используем путь /job")
else:
    ROOT_DIR = local_project_root
    print("✓ Обнаружена локальная среда. Используем локальный путь")

# Добавляем возможные пути к модулям в sys.path
possible_paths_to_models = [
    ROOT_DIR / "models",  # Папка models в корне
    ROOT_DIR / "src" / "models",  # Папка models внутри src
    ROOT_DIR,  # Сам корень проекта
    ROOT_DIR / "src",  # Папка src
    ROOT_DIR / "utils",  # Папка utils в корне
    ROOT_DIR / "src" / "utils",  # Папка utils внутри src
]

for path in possible_paths_to_models:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"✓ Добавлен путь: {path}")

# Также добавляем родительскую директорию текущего файла
current_parent = str(current_file.parent)
if current_parent not in sys.path:
    sys.path.insert(0, current_parent)

print("=" * 60)
print("FINAL ENVIRONMENT INFO:")
print(f"Current file: {current_file}")
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"Current working directory: {Path.cwd()}")
print(f"Python will look for modules in:")
for i, path in enumerate(sys.path[:10]):  # Показываем первые 10 путей
    print(f"  {i+1}. {path}")
print("=" * 60)

# Диагностика: что действительно есть в облаке
print("\nCHECKING CLOUD ENVIRONMENT STRUCTURE:")
check_paths = [ROOT_DIR, Path(".")]
for path in check_paths:
    if path.exists():
        print(f"\nСодержимое {path}:")
        try:
            items = list(path.iterdir())
            if not items:
                print("  [EMPTY]")
            for item in items:
                item_type = "DIR" if item.is_dir() else "FILE"
                print(f"  [{item_type}] {item.name}")
        except Exception as e:
            print(f"  Ошибка доступа: {e}")
print("=" * 60)

from data.faiss_indexer import FaissIndexer
from models.feature_extractor import FeatureExtractor
from utils.config import DATA_PATHS, s3_manager


def create_directories():
    """Создание необходимых директорий"""
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/index", exist_ok=True)
    print("Директории созданы")


def validate_s3_connection():
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


def main():
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
    with open("data/processed/raw_features.pkl", "wb") as f:
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
        with open("data/processed/failed_images.txt", "w") as f:
            for img in failed_images:
                f.write(f"{img}\n")
        print(f"Список неудачных изображений сохранен: data/processed/failed_images.txt")


if __name__ == "__main__":
    main()
