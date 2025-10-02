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
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import pickle

import faiss
import numpy as np
from tqdm import tqdm

from src.utils.config import FAISS_CONFIG


class FaissIndexer:
    """FAISS индекс для поиска похожих изображений по эмбеддингам"""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = None
        self.mapping = {}  # index_id -> metadata
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

            # Загружаем маппинг
            if os.path.exists(mapping_path):
                df = pd.read_csv(mapping_path)
                for _, row in df.iterrows():
                    index_id = int(row["index_id"])
                    self.mapping[index_id] = {"s3_key": row["s3_key"], "lat": row.get("lat"), "lon": row.get("lon")}
                    self.reverse_mapping[row["s3_key"]] = index_id
                logger.info(f"Загружено {len(self.mapping)} записей маппинга")
            else:
                logger.warning(f"Файл маппинга не найден: {mapping_path}")

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
                    metadata = self.mapping[idx]
                    results.append(
                        {
                            "s3_key": metadata["s3_key"],
                            "similarity": float(score),
                            "rank": i + 1,
                            "lat": metadata.get("lat"),
                            "lon": metadata.get("lon"),
                        }
                    )

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

    def get_index_size(self) -> int:
        """Возвращает размер индекса"""
        return self.index.ntotal if self.index else 0
