import os

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

from utils.s3_optimize import S3Manager

s3_manager = S3Manager(
    key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    bucket_name=os.getenv("AWS_BUCKET_NAME"),
    max_workers=32,
    chunk_size=8 * 1024 * 1024,
)

# Настройки модели
MODEL_CONFIG = {"model_name": "ResNet50", "input_size": (224, 224), "pooling": "avg"}

# Настройки FAISS
FAISS_CONFIG = {"index_type": "IVF2048,Flat", "nlist": 2048, "metric": "l2"}

# Пути для сохранения индекса (локально)
DATA_PATHS = {
    "faiss_index": "data/index/faiss_index.bin",
    "mapping_file": "data/processed/image_mapping.pkl",
    "metadata_file": "data/processed/metadata.json",
}

# Настройки производительности
PERFORMANCE_CONFIG = {
    "max_concurrent_tasks": 4,
    "batch_size": 32,
    "max_image_size": (1920, 1080),
    "cache_features": True,
    "enable_gpu": False,
    "processing_timeout": 300,  # 5 минут на обработку одного изображения
}
