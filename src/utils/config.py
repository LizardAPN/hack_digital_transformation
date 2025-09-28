import os

from utils.s3_optimize import S3Manager

s3_manager = S3Manager(
    key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    bucket_name=os.getenv("AWS_BUCKET_NAME"),
    max_workers=4,
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
