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
    """Класс для работы с FAISS индексом для поиска похожих изображений"""

    def __init__(self, dimension: int = 2048) -> None:
        """
        Инициализация FAISS индекса

        Parameters
        ----------
        dimension : int, optional
            Размерность вектора признаков (по умолчанию 2048)
        """
        self.dimension: int = dimension
        self.index: Optional[faiss.Index] = None
        self.image_mapping: Dict[int, Dict[str, Any]] = {}

    def create_index(self, features_dict: Dict[str, Dict[str, Any]], index_type: str = "IVF") -> int:
        """
        Создание FAISS индекса из признаков

        Parameters
        ----------
        features_dict : Dict[str, Dict[str, Any]]
            Словарь признаков {s3_key: {"features": np.ndarray, ...}}
        index_type : str, optional
            Тип индекса ("Flat" или "IVF") (по умолчанию "IVF")

        Returns
        -------
        int
            Количество проиндексированных изображений

        Examples
        --------
        >>> indexer = FaissIndexer(dimension=2048)
        >>> features_dict = {"image1.jpg": {"features": np.random.rand(2048)}}
        >>> num_indexed = indexer.create_index(features_dict, index_type="IVF")
        >>> print(f"Проиндексировано изображений: {num_indexed}")
        """
        s3_keys: List[str] = []
        features_list: List[np.ndarray] = []

        print("Подготовка данных для FAISS...")
        for i, (s3_key, data) in enumerate(tqdm(features_dict.items())):
            s3_keys.append(s3_key)
            features_list.append(data["features"])

            self.image_mapping[i] = {"s3_key": s3_key, "features": data["features"]}

        features_matrix = np.array(features_list).astype("float32")
        print(f"Размер матрицы признаков: {features_matrix.shape}")

        # Создание индекса
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, FAISS_CONFIG["nlist"], faiss.METRIC_L2)

            print("Обучение FAISS индекса...")
            self.index.train(features_matrix)

        print("Добавление данных в индекс...")
        self.index.add(features_matrix)

        return len(features_matrix)

    def search_similar(self, query_features: np.ndarray, k: int = 10) -> List[Dict[str, Union[int, str, float]]]:
        """
        Поиск k наиболее похожих изображений

        Parameters
        ----------
        query_features : np.ndarray
            Вектор признаков для поиска
        k : int, optional
            Количество похожих изображений для возврата (по умолчанию 10)

        Returns
        -------
        List[Dict[str, Union[int, str, float]]]
            Список результатов поиска

        Examples
        --------
        >>> indexer = FaissIndexer(dimension=2048)
        >>> query_features = np.random.rand(2048)
        >>> results = indexer.search_similar(query_features, k=5)
        >>> for result in results:
        ...     print(f"Ранг: {result['rank']}, Расстояние: {result['distance']}")
        """
        if self.index is None:
            raise ValueError("Индекс не инициализирован")

        query_features = query_features.astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_features, k)

        results: List[Dict[str, Union[int, str, float]]] = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.image_mapping:
                results.append(
                    {
                        "rank": i + 1,
                        "s3_key": self.image_mapping[idx]["s3_key"],
                        "distance": float(distance),
                        "similarity_score": 1 / (1 + distance),
                        "index_id": int(idx),
                    }
                )

        return results

    def save_index(self, index_path: str, mapping_path: str) -> None:
        """
        Сохранение индекса и маппинга

        Parameters
        ----------
        index_path : str
            Путь для сохранения индекса
        mapping_path : str
            Путь для сохранения маппинга

        Examples
        --------
        >>> indexer = FaissIndexer(dimension=2048)
        >>> indexer.create_index(features_dict)
        >>> indexer.save_index("data/index/faiss_index.bin", "data/processed/image_mapping.pkl")
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, index_path)

        with open(mapping_path, "wb") as f:
            pickle.dump(self.image_mapping, f)

        print(f"Индекс сохранен: {index_path}")
        print(f"Маппинг сохранен: {mapping_path}")

    def load_index(self, index_path: str, mapping_path: str) -> None:
        """
        Загрузка индекса и маппинга

        Parameters
        ----------
        index_path : str
            Путь к сохраненному индексу
        mapping_path : str
            Путь к сохраненному маппингу

        Examples
        --------
        >>> indexer = FaissIndexer(dimension=2048)
        >>> indexer.load_index("data/index/faiss_index.bin", "data/processed/image_mapping.pkl")
        >>> print(f"Размер загруженного индекса: {indexer.index.ntotal}")
        """
        self.index = faiss.read_index(index_path)

        with open(mapping_path, "rb") as f:
            self.image_mapping = pickle.load(f)

        print(f"Индекс загружен: {index_path}")
        if self.index is not None:
            print(f"Размер индекса: {self.index.ntotal}")
