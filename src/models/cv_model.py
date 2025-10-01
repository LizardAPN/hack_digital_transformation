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
import torchvision.transforms as transforms

from src.geo.geocoder import geocode_coordinates
from src.utils.config import DATA_PATHS, s3_manager

logger = logging.getLogger(__name__)


class FaissIndexer:
    """FAISS индекс для поиска похожих изображений"""
    
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = None
        self.mapping = {}  # index_id -> s3_key (полный путь)
        self.id_mapping = {}  # index_id -> image_id (только ID)
    
    def load_index(self, index_path: str, mapping_path: str):
        """Загружает FAISS индекс и маппинг"""
        try:
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"FAISS индекс загружен: {index_path}, размер: {self.index.ntotal} векторов")
            else:
                logger.warning(f"FAISS индекс не найден: {index_path}")
                return
            
            # Загружаем маппинг
            if os.path.exists(mapping_path):
                df = pd.read_csv(mapping_path)
                logger.info(f"Колонки в файле маппинга: {df.columns.tolist()}")
                
                # Проверяем наличие необходимых колонок
                if 'index_id' in df.columns and 's3_key' in df.columns and 'image_id' in df.columns:
                    for _, row in df.iterrows():
                        idx = row['index_id']
                        self.mapping[idx] = row['s3_key']  # Полный путь в S3
                        self.id_mapping[idx] = row['image_id']  # Только ID
                    logger.info(f"Загружено {len(self.mapping)} записей маппинга")
                else:
                    logger.error(f"В файле маппинга отсутствуют необходимые колонки")
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки FAISS индекса: {e}")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Поиск k ближайших соседей"""
        if self.index is None:
            logger.error("FAISS индекс не загружен")
            return []
        
        try:
            # Проверяем размерность запроса
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            logger.info(f"Размерность запроса: {query_embedding.shape}")
            logger.info(f"Размерность индекса: {self.index.d}, количество векторов: {self.index.ntotal}")
            
            # Нормализуем запрос для косинусного расстояния
            query_norm = self._l2_normalize(query_embedding)
            
            # Поиск в индексе
            scores, indices = self.index.search(query_norm, k)
            
            logger.info(f"Результаты поиска: scores={scores}, indices={indices}")
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx in self.mapping and idx in self.id_mapping:
                    results.append({
                        "s3_key": self.mapping[idx],  # Полный путь в S3
                        "image_id": self.id_mapping[idx],  # Только ID
                        "similarity": float(score),
                        "rank": i + 1
                    })
                else:
                    logger.warning(f"Индекс {idx} не найден в маппинге")
            
            logger.info(f"Найдено результатов: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска в FAISS: {e}", exc_info=True)
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
        
        # Создаем трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Инициализация FAISS индексера
        self.faiss_indexer = FaissIndexer(dimension=512)
        self.faiss_indexer.load_index(faiss_index_path, mapping_path)
        
        # Загружаем метаданные (координаты по image_id)
        self.metadata = self._load_metadata(train_metadata_path)
        logger.info(f"Загружено {len(self.metadata)} записей метаданных")

    def _load_metadata(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Загружает метаданные: image_id -> {lat, lon}"""
        metadata = {}
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(f"Колонки в файле метаданных: {df.columns.tolist()}")
                
                # Используем колонку 'id' как image_id
                for _, row in df.iterrows():
                    try:
                        image_id = str(row['id'])
                        lat = float(row['latitude'])
                        lon = float(row['longitude'])
                        metadata[image_id] = {"lat": lat, "lon": lon}
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Пропуск строки с невалидными данными: {e}")
                        continue
                
                logger.info(f"Метаданные загружены из {csv_path}")
            else:
                logger.warning(f"Файл метаданных не найден: {csv_path}")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
        
        return metadata

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Кодирует изображение в эмбеддинг с помощью GeoCLIP"""
        try:
            # Используем тот же препроцессинг, что и при создании индекса
            tensor = self.transform(image)
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
                image_id = best_match["image_id"]  # Используем ID для поиска в метаданных
                similarity = best_match["similarity"]
                
                logger.info(f"Лучшее совпадение: image_id={image_id} (сходство: {similarity:.4f})")

                # Получаем координаты по ID
                coords = self.metadata.get(image_id)
                if not coords:
                    logger.warning(f"Нет координат для изображения: {image_id}, используем тестовые координаты")
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