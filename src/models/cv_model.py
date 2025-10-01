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
import os 
import numpy as np
import pandas as pd
from PIL import Image
import faiss
from geoclip import ImageEncoder
import torch

from src.geo.geocoder import geocode_coordinates
from src.utils.config import DATA_PATHS, s3_manager

logger = logging.getLogger(__name__)


class FaissIndexer:
    """FAISS индекс для поиска похожих изображений"""
    
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = None
        self.mapping = {}  # index_id -> s3_key
        self.reverse_mapping = {}  # s3_key -> index_id
    
    def load_index(self, index_path: str, mapping_path: str):
        """Загружает FAISS индекс и маппинг"""
        try:
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"FAISS индекс загружен: {index_path}")
            else:
                logger.warning(f"FAISS индекс не найден: {index_path}")
                return
            
            # Загружаем маппинг с проверкой наличия колонок
            if os.path.exists(mapping_path):
                df = pd.read_csv(mapping_path)
                logger.info(f"Колонки в файле маппинга: {df.columns.tolist()}")
                
                # Проверяем наличие необходимых колонок
                if 'index_id' in df.columns and 's3_key' in df.columns:
                    for _, row in df.iterrows():
                        self.mapping[row['index_id']] = row['s3_key']
                        self.reverse_mapping[row['s3_key']] = row['index_id']
                    logger.info(f"Загружено {len(self.mapping)} записей маппинга")
                else:
                    logger.error(f"В файле маппинга отсутствуют необходимые колонки. Найдены: {df.columns.tolist()}")
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки FAISS индекса: {e}")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Поиск k ближайших соседей"""
        if self.index is None:
            logger.error("FAISS индекс не загружен")
            return []
        
        try:
            # Нормализуем запрос для косинусного расстояния
            query_norm = self._l2_normalize(query_embedding)
            
            # Поиск в индексе
            scores, indices = self.index.search(query_norm, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx in self.mapping:
                    results.append({
                        "s3_key": self.mapping[idx],
                        "similarity": float(score),
                        "rank": i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска в FAISS: {e}")
            return []
    
    def _l2_normalize(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """L2 нормализация векторов"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        return x / np.maximum(norm, eps)


class CVModel:
    """Модель компьютерного зрения на основе GeoCLIP с поиском по FAISS"""

    def __init__(
        self,
        faiss_index_path: str = "data/index/faiss_index.bin",
        mapping_path: str = "data/index/image_mapping.csv",
        train_metadata_path: str = "data/processed_data/moscow_images.csv"
    ):
        """
        Инициализация модели: загрузка GeoCLIP и FAISS индекса.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")
        
        # Инициализация GeoCLIP
        self.image_encoder = ImageEncoder().to(self.device).eval()
        logger.info("GeoCLIP модель инициализирована")
        
        # Инициализация FAISS индексера
        self.faiss_indexer = FaissIndexer(dimension=512)
        self.faiss_indexer.load_index(faiss_index_path, mapping_path)
        
        # Загружаем метаданные (координаты по s3_key)
        self.metadata = self._load_metadata(train_metadata_path)
        logger.info(f"Загружено {len(self.metadata)} записей метаданных")

    def _load_metadata(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Загружает метаданные: s3_key -> {lat, lon}"""
        metadata = {}
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(f"Колонки в файле метаданных: {df.columns.tolist()}")
                
                # Определяем названия колонок
                s3_key_col = None
                lat_col = None
                lon_col = None
                
                # Ищем подходящие колонки
                for col in df.columns:
                    col_lower = col.lower()
                    if 'key' in col_lower or 'path' in col_lower or 'file' in col_lower:
                        s3_key_col = col
                    elif 'lat' in col_lower:
                        lat_col = col
                    elif 'lon' in col_lower:
                        lon_col = col
                
                # Если не нашли автоматически, используем первые три колонки
                if not s3_key_col and len(df.columns) >= 3:
                    s3_key_col = df.columns[0]
                    lat_col = df.columns[1]
                    lon_col = df.columns[2]
                    logger.info(f"Используем колонки по умолчанию: {s3_key_col}, {lat_col}, {lon_col}")
                
                if s3_key_col and lat_col and lon_col:
                    for _, row in df.iterrows():
                        try:
                            s3_key = str(row[s3_key_col])
                            lat = float(row[lat_col])
                            lon = float(row[lon_col])
                            metadata[s3_key] = {"lat": lat, "lon": lon}
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Пропуск строки с невалидными данными: {row}")
                            continue
                    
                    logger.info(f"Метаданные загружены из {csv_path}")
                else:
                    logger.error(f"Не удалось определить колонки в файле метаданных. Доступные колонки: {df.columns.tolist()}")
                    
            else:
                logger.warning(f"Файл метаданных не найден: {csv_path}")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
        
        return metadata

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Кодирует изображение в эмбеддинг с помощью GeoCLIP"""
        try:
            # Предобработка изображения
            tensor = self.image_encoder.preprocess_image(image)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
            
            # Перенос на устройство
            tensor = tensor.to(self.device)
            
            # Получение эмбеддинга
            with torch.no_grad():
                embedding = self.image_encoder(tensor)
                embedding = embedding.cpu().numpy()
            
            logger.info(f"Получен эмбеддинг формы: {embedding.shape}")
            return embedding.astype("float32")
            
        except Exception as e:
            logger.error(f"Ошибка кодирования изображения: {e}")
            raise

    def process_image(self, image_path: str) -> Dict:
        """
        Обработка изображения: определение координат через GeoCLIP + FAISS
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

            # Загрузка изображения из S3
            logger.info(f"Загрузка изображения из S3: {image_path}")
            image_data = s3_manager.download_bytes(image_path)
            if image_data is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение из S3: {image_path}")
            
            # Открываем изображение
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            logger.info(f"Изображение загружено: {image.size}")

            # Кодирование в эмбеддинг
            query_embedding = self._encode_image(image)

            # Поиск похожих изображений
            similar_results = self.faiss_indexer.search_similar(query_embedding, k=3)
            logger.info(f"Найдено похожих изображений: {len(similar_results)}")

            if not similar_results:
                # Если нет похожих изображений, используем заглушку для тестирования
                logger.warning("Не найдено похожих изображений, используем тестовые координаты")
                result["coordinates"] = {"lat": 55.7558, "lon": 37.6173}  # Москва
            else:
                # Берем самое похожее изображение
                best_match = similar_results[0]
                s3_key = best_match["s3_key"]
                similarity = best_match["similarity"]
                
                logger.info(f"Лучшее совпадение: {s3_key} (сходство: {similarity:.4f})")

                # Получаем координаты
                coords = self.metadata.get(s3_key)
                if not coords:
                    logger.warning(f"Нет координат для изображения: {s3_key}, используем тестовые координаты")
                    result["coordinates"] = {"lat": 55.7558, "lon": 37.6173}
                else:
                    result["coordinates"] = coords
                    logger.info(f"Определены координаты: {coords}")

            # Геокодирование
            if result["coordinates"]:
                try:
                    address = geocode_coordinates(result["coordinates"]["lat"], result["coordinates"]["lon"])
                    result["address"] = address
                    logger.info(f"Определен адрес: {address}")
                except Exception as e:
                    logger.warning(f"Ошибка геокодирования: {e}")
                    result["address"] = None

            # Детекция зданий (заглушка)
            result["buildings"] = self._detect_buildings_placeholder()

            # OCR (заглушка)
            result["ocr_result"] = self._perform_ocr_placeholder()

            return result

        except Exception as e:
            logger.error(f"Ошибка обработки изображения {image_path}: {e}")
            raise

    def _detect_buildings_placeholder(self) -> List[Dict]:
        """Заглушка для детекции зданий"""
        return [{
            "bbox": [0, 0, 100, 100], 
            "confidence": 1.0, 
            "area": 10000
        }]

    def _perform_ocr_placeholder(self) -> Dict:
        """Заглушка для OCR"""
        return {
            "text": "ТЕКСТ НЕ РАСПОЗНАН",
            "confidence": 0.0,
            "roi_name": "full_image"
        }


def create_cv_model() -> CVModel:
    """
    Фабричная функция для создания экземпляра CVModel
    """
    return CVModel()