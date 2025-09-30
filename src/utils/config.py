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
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Определяем корневую директорию проекта
# Используем переменную окружения PROJECT_ROOT если задана, иначе определяем автоматически
PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.parent)).resolve()

# Пути к основным директориям
SRC_DIR: Path = PROJECT_ROOT / "src"
MODELS_DIR: Path = SRC_DIR / "models"
UTILS_DIR: Path = SRC_DIR / "utils"
DATA_DIR: Path = PROJECT_ROOT / "data"

# Создаем директории если они не существуют
for directory in [SRC_DIR, MODELS_DIR, UTILS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

from src.utils.s3_optimize import S3Manager

s3_manager: Optional[S3Manager] = None

try:
    s3_manager = S3Manager(
        key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        bucket_name=os.getenv("AWS_BUCKET_NAME"),
        max_workers=32,
        chunk_size=8 * 1024 * 1024,
    )
except ValueError:
    # В тестовой среде без учетных данных AWS создаем заглушку
    s3_manager = None

# Настройки модели
MODEL_CONFIG: Dict[str, Any] = {"model_name": "ResNet50", "input_size": (224, 224), "pooling": "avg"}

# Настройки FAISS
FAISS_CONFIG: Dict[str, Any] = {"index_type": "IVF2048,Flat", "nlist": 2048, "metric": "l2"}

# Пути для сохранения индекса (локально)
DATA_PATHS: Dict[str, str] = {
    "faiss_index": "data/index/faiss_index.bin",
    "mapping_file": "data/processed/image_mapping.pkl",
    "metadata_file": "data/processed/metadata.json",
}

# Настройки производительности
PERFORMANCE_CONFIG: Dict[str, Any] = {
    "max_concurrent_tasks": 4,
    "batch_size": 32,
    "max_image_size": (1920, 1080),
    "cache_features": True,
    "enable_gpu": False,
    "processing_timeout": 300,  # 5 минут на обработку одного изображения
}
