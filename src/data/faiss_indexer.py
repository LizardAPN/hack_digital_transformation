import os
import pickle

import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root_in_cloud = Path("/job")  # Явно указываем корень в облаке
local_project_root = current_file.parent.parent

# Выбираем корень в зависимости от окружения
# Проверяем, находимся ли мы в среде DataSphere (существует ли папка /job)
if project_root_in_cloud.exists():
    ROOT_DIR = project_root_in_cloud
    print("✓ Обнаружена среда DataSphere. Используем путь /job")
else:
    ROOT_DIR = local_project_root
    print("✓ Обнаружена локальная среда. Используем локальный путь")

# Добавляем возможные пути к модулям в sys.path
possible_paths_to_models = [
    ROOT_DIR / "models",  # Папка models в корне
    ROOT_DIR / "src" / "models",  # Папка models внутри src
    ROOT_DIR,  # Сам корень проекта
    ROOT_DIR / "src",  # Папка src
    ROOT_DIR / "utils",  # Папка utils в корне
    ROOT_DIR / "src" / "utils",  # Папка utils внутри src
]

for path in possible_paths_to_models:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"✓ Добавлен путь: {path}")

# Также добавляем родительскую директорию текущего файла
current_parent = str(current_file.parent)
if current_parent not in sys.path:
    sys.path.insert(0, current_parent)

print("=" * 60)
print("FINAL ENVIRONMENT INFO:")
print(f"Current file: {current_file}")
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"Current working directory: {Path.cwd()}")
print(f"Python will look for modules in:")
for i, path in enumerate(sys.path[:10]):  # Показываем первые 10 путей
    print(f"  {i+1}. {path}")
print("=" * 60)

# Диагностика: что действительно есть в облаке
print("\nCHECKING CLOUD ENVIRONMENT STRUCTURE:")
check_paths = [ROOT_DIR, Path(".")]
for path in check_paths:
    if path.exists():
        print(f"\nСодержимое {path}:")
        try:
            items = list(path.iterdir())
            if not items:
                print("  [EMPTY]")
            for item in items:
                item_type = "DIR" if item.is_dir() else "FILE"
                print(f"  [{item_type}] {item.name}")
        except Exception as e:
            print(f"  Ошибка доступа: {e}")
print("=" * 60)

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
