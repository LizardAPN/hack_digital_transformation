import json
import os
import pickle

import faiss

from pathlib import Path
import sys

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
