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


class CVModel:
    """Модель компьютерного зрения для детекции зданий и определения координат на основе GeoCLIP"""

    def __init__(self):
        """
        Инициализация модели CV с использованием GeoCLIP

        Examples
        --------
        >>> model = CVModel()
        >>> result = model.process_image("path/to/image.jpg")
        """
        # Инициализируем GeoCLIP модель
        self.image_encoder = ImageEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_encoder = self.image_encoder.to(self.device)
        self.image_encoder.eval()

    def process_image(self, image_path: str) -> Dict:
        """
        Обработка изображения: определение координат с помощью GeoCLIP

        Parameters
        ----------
        image_path : str
            Путь к изображению

        Returns
        -------
        Dict
            Словарь с результатами обработки

        Examples
        --------
        >>> model = CVModel()
        >>> result = model.process_image("path/to/image.jpg")
        >>> print(result["coordinates"])
        """
        try:
            result = {
                "image_path": image_path,
                "processed_at": datetime.now().isoformat(),
                "buildings": [],
                "coordinates": None,
                "address": None,
                "ocr_result": None,
            }

            # Загружаем изображение из S3
            image_data = s3_manager.download_bytes(image_path)
            if image_data is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение из S3: {image_path}")
            
            # Открываем изображение из байтов
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Предобрабатываем изображение для GeoCLIP
            preprocessed_image = self.image_encoder.preprocess_image(image)
            if preprocessed_image.ndim == 3:
                preprocessed_image = preprocessed_image.unsqueeze(0)
            
            # Переносим изображение на устройство
            preprocessed_image = preprocessed_image.to(self.device)

            # Получаем предсказания координат от GeoCLIP
            with torch.no_grad():
                pred_coords = self.image_encoder(preprocessed_image)
                # pred_coords имеет форму [1, 2] с широтой и долготой
                lat, lon = pred_coords[0].cpu().numpy()
            
            # Сохраняем координаты в результате
            result["coordinates"] = {"lat": float(lat), "lon": float(lon)}

            # Получаем адрес по координатам
            try:
                address = geocode_coordinates(lat, lon)
                result["address"] = address
            except Exception as e:
                logger.warning(f"Ошибка геокодирования координат {lat}, {lon}: {e}")

            # Детектируем здания (заглушка, так как у нас GeoCLIP)
            result["buildings"] = [{"bbox": [0, 0, 100, 100], "confidence": 1.0, "area": 10000}]  # Заглушка

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
