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
import logging
from datetime import datetime
from typing import Dict, List, Optional
import io

import numpy as np
from PIL import Image

from geoclip import ImageEncoder
import torch

from src.geo.geocoder import geocode_coordinates
from src.utils.config import DATA_PATHS, s3_manager

logger = logging.getLogger(__name__)


from src.data.faiss_indexer import FaissIndexer  # Импортируем ваш индексер

logger = logging.getLogger(__name__)

class CVModel:
    """Модель компьютерного зрения на основе GeoCLIP с поиском по FAISS"""

    def __init__(
        self,
        faiss_index_path: str = "data/index/geoclip_faiss.bin",
        mapping_path: str = "data/index/geoclip_mapping.pkl",
        train_metadata_path: str = "data/train_metadata.csv"
    ):
        """
        Инициализация модели: загрузка GeoCLIP и FAISS индекса.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_encoder = ImageEncoder().to(self.device).eval()

        # Инициализируем FAISS
        self.faiss_indexer = FaissIndexer(dimension=512)  # Размер эмбеддинга GeoCLIP
        self.faiss_indexer.load_index(faiss_index_path, mapping_path)

        # Загружаем метаданные (координаты по ID)
        self.metadata = self._load_metadata(train_metadata_path)

    def _load_metadata(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Загружает метаданные: s3_key -> {lat, lon}"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        metadata = {}
        for _, row in df.iterrows():
            metadata[row["s3_key"]] = {"lat": row["lat"], "lon": row["lon"]}
        return metadata

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Кодирует изображение в эмбеддинг"""
        tensor = self.image_encoder.preprocess_image(image)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        tensor = tensor.to(self.device)
        with torch.no_grad():
            emb = self.image_encoder(tensor).cpu().numpy()
        return emb.astype("float32")

    def process_image(self, image_path: str, is_local: bool = False) -> Dict:
        try:
            result = {
                "image_path": image_path,
                "processed_at": datetime.now().isoformat(),
                "buildings": [],
                "coordinates": None,
                "address": None,
                "ocr_result": None,
            }

            # Загрузка изображения
            if is_local:
                image = Image.open(image_path).convert("RGB")
            else:
                image_data = s3_manager.download_bytes(image_path)
                if image_data is None:
                    raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
                image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Кодирование
            query_emb = self._encode_image(image)

            # Поиск ближайшего
            results = self.faiss_indexer.search_similar(query_emb, k=1)
            if not results:
                raise ValueError("Не найдено ближайших изображений")

            nearest_s3_key = results[0]["s3_key"]
            coords = self.metadata.get(nearest_s3_key)
            if not coords:
                raise ValueError(f"Нет координат для {nearest_s3_key}")

            result["coordinates"] = {"lat": coords["lat"], "lon": coords["lon"]}

            # Геокодирование
            try:
                address = geocode_coordinates(coords["lat"], coords["lon"])
                result["address"] = address
            except Exception as e:
                logger.warning(f"Ошибка геокодирования: {e}")

            # Заглушка для зданий
            result["buildings"] = [{"bbox": [0, 0, 100, 100], "confidence": 1.0, "area": 10000}]

            return result

        except Exception as e:
            logger.error(f"Ошибка обработки изображения {image_path}: {e}")
            raise


def create_cv_model() -> CVModel:
    """
    Фабричная функция для создания экземпляра CVModel

    Returns
    -------
    CVModel
        Экземпляр CVModel

    Examples
    --------
    >>> model = create_cv_model()
    >>> result = model.process_image("path/to/image.jpg")
    """
    return CVModel()
