import io

import matplotlib.pyplot as plt
import torch
from PIL import Image

from data.faiss_indexer import FaissIndexer
from models.feature_extractor import FeatureExtractor
from utils.config import DATA_PATHS, s3_manager


def test_search():
    """Тестирование поиска по базе"""
    print("=== Тестирование поиска похожих изображений ===")

    # Загрузка индекса
    print("Загрузка FAISS индекса...")
    indexer = FaissIndexer()
    indexer.load_index(DATA_PATHS["faiss_index"], DATA_PATHS["mapping_file"])

    # Загрузка feature extractor
    print("Инициализация feature extractor...")
    extractor = FeatureExtractor()

    # Тестовое изображение (можно заменить на любой файл)
    test_image_path = "/home/lizardapn/Hack_digital/hack_digital_transformation/data/processed_data/moscow_image/1850854291755864.jpeg"
    try:
        test_image = Image.open(test_image_path)
        if test_image.mode != "RGB":
            test_image = test_image.convert("RGB")
        print(f"Тестовое изображение загружено: {test_image_path}")
    except Exception as e:
        print(f"Ошибка загрузки тестового изображения: {e}")
        # Альтернатива: использовать случайное изображение из базы
        if indexer.image_mapping:
            first_key = list(indexer.image_mapping.values())[0]["s3_key"]
            image_data = s3_manager.download_bytes(first_key)
            if image_data:
                test_image = Image.open(io.BytesIO(image_data))
                print(f"Используется первое изображение из базы: {first_key}")
            else:
                print("Не удалось загрузить тестовое изображение")
                return
        else:
            print("База данных пуста")
            return

    # Извлечение признаков
    print("Извлечение признаков из тестового изображения...")
    features = extractor.extract_features(test_image)

    if features is None:
        print("Не удалось извлечь признаки из тестового изображения")
        return

    # Поиск похожих
    print("Поиск похожих изображений...")
    results = indexer.search_similar(features, k=5)

    print("\n" + "=" * 50)
    print("ТОП-5 ПОХОЖИХ ИЗОБРАЖЕНИЙ:")
    print("=" * 50)

    for i, result in enumerate(results):
        print(f"{result['rank']}. {result['s3_key']}")
        print(f"   Схожесть: {result['similarity_score']:.3f}")
        print(f"   Расстояние: {result['distance']:.3f}")
        print()

    # Визуализация результатов
    try:
        visualize_results(test_image, results, extractor)
    except Exception as e:
        print(f"Ошибка визуализации: {e}")


def visualize_results(query_image, results, extractor):
    """Визуализация результатов поиска"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Запрос
    axes[0].imshow(query_image)
    axes[0].set_title("Запрос", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Результаты
    for i, result in enumerate(results[:5]):
        similar_image_data = s3_manager.download_bytes(result["s3_key"])
        if similar_image_data:
            similar_image = Image.open(io.BytesIO(similar_image_data))
            axes[i + 1].imshow(similar_image)
            axes[i + 1].set_title(f"#{i+1} Схожесть: {result['similarity_score']:.3f}\n{result['s3_key']}", fontsize=10)
            axes[i + 1].axis("off")
        else:
            axes[i + 1].text(
                0.5,
                0.5,
                f"Не удалось\nзагрузить\n{result['s3_key']}",
                ha="center",
                va="center",
                transform=axes[i + 1].transAxes,
            )
            axes[i + 1].axis("off")

    # Скрываем пустые subplots
    for i in range(len(results) + 1, 6):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_search()
