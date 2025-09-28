import os
import pickle

import faiss
import numpy as np
from tqdm import tqdm

from utils.config import FAISS_CONFIG


class FaissIndexer:
    def __init__(self, dimension=2048):
        self.dimension = dimension
        self.index = None
        self.image_mapping = {}

    def create_index(self, features_dict, index_type="IVF"):
        """Создание FAISS индекса из признаков"""
        s3_keys = []
        features_list = []

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

    def search_similar(self, query_features, k=10):
        """Поиск k наиболее похожих изображений"""
        if self.index is None:
            raise ValueError("Индекс не инициализирован")

        query_features = query_features.astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_features, k)

        results = []
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

    def save_index(self, index_path, mapping_path):
        """Сохранение индекса и маппинга"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

        faiss.write_index(self.index, index_path)

        with open(mapping_path, "wb") as f:
            pickle.dump(self.image_mapping, f)

        print(f"Индекс сохранен: {index_path}")
        print(f"Маппинг сохранен: {mapping_path}")

    def load_index(self, index_path, mapping_path):
        """Загрузка индекса и маппинга"""
        self.index = faiss.read_index(index_path)

        with open(mapping_path, "rb") as f:
            self.image_mapping = pickle.load(f)

        print(f"Индекс загружен: {index_path}")
        print(f"Размер индекса: {self.index.ntotal}")
