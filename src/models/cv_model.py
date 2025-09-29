import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Импортируем существующие компоненты
from .feature_extractor import FeatureExtractor
from .OCR_model import OverlayOCR
from ..data.faiss_indexer import FaissIndexer
from ..geo.geocoder import geocode_coordinates
from ..utils.config import DATA_PATHS, s3_manager

logger = logging.getLogger(__name__)


class CVModel:
    """Модель компьютерного зрения для детекции зданий и определения координат"""

    def __init__(self):
        """Инициализация модели CV"""
        self.ocr_model = OverlayOCR()
        self.feature_extractor = FeatureExtractor()
        self.indexer = None
        self._initialize_faiss_index()

    def _initialize_faiss_index(self):
        """Инициализация FAISS индекса"""
        try:
            logger.info("Инициализация FAISS индекса...")
            self.indexer = FaissIndexer(dimension=2048)
            self.indexer.load_index(DATA_PATHS["faiss_index"], DATA_PATHS["mapping_file"])
            logger.info("FAISS индекс инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации FAISS индекса: {e}")
            raise

    def process_image(self, image_path: str) -> Dict:
        """
        Обработка изображения: детекция зданий, определение координат, OCR

        Args:
            image_path: Путь к изображению

        Returns:
            Словарь с результатами обработки
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

            # Загружаем изображение
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Извлекаем признаки изображения
            features = self.feature_extractor.extract_features(image)

            if features is not None:
                # Поиск похожих изображений в базе
                similar_images = self.indexer.search_similar(features, k=5)

                # Определяем координаты на основе топ-N похожих изображений
                coordinates = self._estimate_coordinates(similar_images)
                result["coordinates"] = coordinates

                # Получаем адрес по координатам
                if coordinates:
                    result["address"] = self._geocode_coordinates(coordinates)

                # Детектируем здания (заглушка, так как у нас визуальный поиск)
                result["buildings"] = self._detect_buildings_placeholder(image_path)

            # Выполняем OCR
            try:
                final, norm, joined, conf, roi_name = self.ocr_model.run_on_image(image_path)
                result["ocr_result"] = {
                    "final": final,
                    "norm": norm,
                    "joined": joined,
                    "confidence": conf,
                    "roi_name": roi_name,
                }
            except Exception as e:
                logger.warning(f"Ошибка OCR для {image_path}: {e}")

            return result

        except Exception as e:
            logger.error(f"Ошибка обработки изображения {image_path}: {e}")
            raise

    def _estimate_coordinates(self, similar_images: List[Dict]) -> Optional[Dict]:
        """
        Оценка координат на основе похожих изображений

        Args:
            similar_images: Список похожих изображений с результатами поиска

        Returns:
            Словарь с координатами {lat, lon} или None
        """
        if not similar_images:
            return None

        try:
            # Получаем координаты из метаданных похожих изображений
            coordinates_list = []

            for img_result in similar_images:
                s3_key = img_result["s3_key"]
                # Здесь нужно извлечь координаты из метаданных S3 объекта
                # или из отдельной базы данных с координатами
                coords = self._get_image_coordinates_from_metadata(s3_key)
                if coords:
                    # Взвешиваем координаты по степени схожести
                    weight = 1.0 / (1.0 + img_result["distance"])
                    coordinates_list.append({"lat": coords["lat"], "lon": coords["lon"], "weight": weight})

            if not coordinates_list:
                return None

            # Вычисляем взвешенное среднее координат
            total_weight = sum(coord["weight"] for coord in coordinates_list)
            if total_weight == 0:
                return None

            avg_lat = sum(coord["lat"] * coord["weight"] for coord in coordinates_list) / total_weight
            avg_lon = sum(coord["lon"] * coord["weight"] for coord in coordinates_list) / total_weight

            return {"lat": avg_lat, "lon": avg_lon}

        except Exception as e:
            logger.error(f"Ошибка оценки координат: {e}")
            return None

    def _get_image_coordinates_from_metadata(self, s3_key: str) -> Optional[Dict]:
        """
        Получение координат изображения из метаданных S3

        Args:
            s3_key: Ключ объекта в S3

        Returns:
            Словарь с координатами {lat, lon} или None
        """
        try:
            # Получаем метаданные объекта из S3
            metadata = s3_manager.get_object_metadata(s3_key)

            if metadata and "latitude" in metadata and "longitude" in metadata:
                lat = float(metadata["latitude"])
                lon = float(metadata["longitude"])
                return {"lat": lat, "lon": lon}

            return None
        except Exception as e:
            logger.warning(f"Не удалось получить координаты для {s3_key}: {e}")
            return None

    def _geocode_coordinates(self, coordinates: Dict) -> Optional[str]:
        """
        Получение адреса по координатам

        Args:
            coordinates: Словарь с координатами {lat, lon}

        Returns:
            Адрес в виде строки или None
        """
        if coordinates and "lat" in coordinates and "lon" in coordinates:
            try:
                address = geocode_coordinates(coordinates["lat"], coordinates["lon"])
                return address
            except Exception as e:
                logger.warning(f"Ошибка геокодирования координат {coordinates}: {e}")
                return None
        return None

    def _detect_buildings_placeholder(self, image_path: str) -> List[Dict]:
        """
        Заглушка для детекции зданий
        В текущей архитектуре используется визуальный поиск, поэтому детекция не требуется

        Args:
            image_path: Путь к изображению

        Returns:
            Список обнаруженных зданий
        """
        # В текущей архитектуре мы не детектируем здания напрямую,
        # а находим похожие изображения в базе
        return [{"bbox": [0, 0, 100, 100], "confidence": 1.0, "area": 10000}]  # Заглушка


def create_cv_model() -> CVModel:
    """Фабричная функция для создания экземпляра CVModel"""
    return CVModel()
