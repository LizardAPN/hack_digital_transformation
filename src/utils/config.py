import os

from utils.s3_optimize import S3Manager

# Менеджер подключения к хранилищу S3
# Используется для загрузки и скачивания файлов из облачного хранилища
s3_manager = S3Manager(
    key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    bucket_name=os.getenv("AWS_BUCKET_NAME"),
    max_workers=4,
    chunk_size=8 * 1024 * 1024,
)

# Настройки модели компьютерного зрения
# model_name: название предобученной модели из библиотеки Keras
# input_size: размер входного изображения для модели
# pooling: тип пулинга на выходе модели (avg, max)
MODEL_CONFIG = {"model_name": "ResNet50", "input_size": (224, 224), "pooling": "avg"}

# Настройки индекса FAISS для поиска похожих изображений
# index_type: тип индекса FAISS (IVF - inverted file index)
# nlist: количество кластеров для IVF
# metric: метрика расстояния (l2 - евклидово расстояние)
FAISS_CONFIG = {"index_type": "IVF2048,Flat", "nlist": 2048, "metric": "l2"}

# Пути для сохранения индекса и вспомогательных файлов (локально)
# faiss_index: путь к бинарному файлу индекса FAISS
# mapping_file: путь к файлу сопоставления идентификаторов изображений
# metadata_file: путь к файлу метаданных изображений
DATA_PATHS = {
    "faiss_index": "data/index/faiss_index.bin",
    "mapping_file": "data/processed/image_mapping.pkl",
    "metadata_file": "data/processed/metadata.json",
}

# Настройки производительности обработки изображений
# max_concurrent_tasks: максимальное количество параллельных задач
# batch_size: размер пакета для обработки изображений
# max_image_size: максимальный размер изображения (ширина, высота)
# cache_features: флаг кэширования извлеченных признаков
# enable_gpu: флаг использования GPU для вычислений
# processing_timeout: таймаут обработки одного изображения в секундах
PERFORMANCE_CONFIG = {
    "max_concurrent_tasks": 4,
    "batch_size": 32,
    "max_image_size": (1920, 1080),
    "cache_features": True,
    "enable_gpu": False,
    "processing_timeout": 300,  # 5 минут на обработку одного изображения
}
