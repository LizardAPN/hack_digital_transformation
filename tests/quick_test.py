import json
import os
import pickle
import time

from data.faiss_indexer import FaissIndexer
from models.feature_extractor import FeatureExtractor
from utils.config import s3_manager


def quick_build_test(max_files=50):
    """Быстрое тестирование на небольшом подмножестве"""
    print(f"=== БЫСТРОЕ ТЕСТИРОВАНИЕ НА {max_files} ФАЙЛАХ ===")

    # 1. Быстро получаем только несколько файлов
    print("1. Получение списка файлов из S3...")
    all_files = s3_manager.list_files(prefix="", file_extensions=[".jpg", ".jpeg", ".png", ".webp"])

    if not all_files:
        print("Не найдено файлов в S3!")
        return

    # Фильтруем только изображения
    image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files = image_files[:max_files]  # Берем только первые N

    print(f"Будет обработано {len(image_files)} изображений")

    # 2. Создаем директории
    os.makedirs("data_test/processed", exist_ok=True)
    os.makedirs("data_test/index", exist_ok=True)

    # 3. Загружаем и обрабатываем изображения
    print("2. Загрузка и обработка изображений...")
    extractor = FeatureExtractor()
    features_dict = {}

    for i, s3_key in enumerate(image_files):
        print(f"  Обработка {i+1}/{len(image_files)}: {s3_key}")

        # Загружаем одно изображение
        image_data = s3_manager.download_bytes(s3_key)
        if image_data:
            # Преобразуем в изображение
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Извлекаем признаки
            features = extractor.extract_features(image)
            if features is not None:
                features_dict[s3_key] = {"features": features, "s3_key": s3_key}

    print(f"Успешно обработано: {len(features_dict)}/{len(image_files)}")

    if not features_dict:
        print("Не удалось обработать ни одного изображения!")
        return

    # 4. Создаем FAISS индекс
    print("3. Создание FAISS индекса...")
    indexer = FaissIndexer(dimension=2048)
    num_indexed = indexer.create_index(features_dict, index_type="Flat")  # Используем простой индекс

    # 5. Сохраняем результаты
    print("4. Сохранение результатов...")
    indexer.save_index("data_test/index/faiss_index.bin", "data_test/processed/image_mapping.pkl")

    # Сохраняем метаданные
    metadata = {
        "test_files_processed": len(features_dict),
        "total_files_in_test": len(image_files),
        "model_used": "ResNet50",
        "build_time": "quick_test",
        "note": "Быстрый тест на ограниченном наборе данных",
    }

    with open("data_test/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("=== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО! ===")
    print(f"Создана тестовая база с {num_indexed} изображениями")
    print("Файлы сохранены в data_test/")


if __name__ == "__main__":
    quick_build_test(max_files=5000)  # Обработаем только 20 файлов для теста
