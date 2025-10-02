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
    paths_to_add = [src_path, src_path / "utils", src_path / "models"]
    for path in paths_to_add:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

from typing import Dict, List, Tuple, Optional, Any
import io
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from utils.config import MODEL_CONFIG, s3_manager


class FeatureExtractor:
    """Класс для извлечения признаков из изображений с помощью ResNet50"""

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Инициализация экстрактора признаков

        Parameters
        ----------
        device : str, optional
            Устройство для вычислений (cuda/cpu) (по умолчанию None)
        """
        # Определяем устройство
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")

        # Загружаем предобученную ResNet50 с обработкой ошибок
        try:
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as e:
            print(f"Ошибка загрузки весов модели из интернета: {e}")
            print("Попытка загрузить модель без предобученных весов...")
            try:
                self.model = models.resnet50(weights=None)
            except Exception as e2:
                print(f"Ошибка загрузки модели без весов: {e2}")
                raise RuntimeError("Не удалось загрузить модель ResNet50")

        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # Трансформации для изображений
        self.transform = transforms.Compose(
            [
                transforms.Resize(MODEL_CONFIG["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Статистика обработки
        self.stats: Dict[str, int] = {"processed": 0, "failed": 0, "total_time": 0}

    def load_image_from_bytes(self, image_data: bytes) -> Optional[Image.Image]:
        """
        Загрузка изображения из байтов

        Parameters
        ----------
        image_data : bytes
            Байтовые данные изображения

        Returns
        -------
        Image.Image или None
            Изображение PIL или None в случае ошибки

        Examples
        --------
        >>> with open("image.jpg", "rb") as f:
        ...     image_data = f.read()
        >>> image = extractor.load_image_from_bytes(image_data)
        >>> if image is not None:
        ...     print(f"Изображение загружено: {image.size}")
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            print(f"Ошибка загрузки изображения из байтов: {e}")
            return None

    def extract_features(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Извлечение признаков из изображения

        Parameters
        ----------
        image : Image.Image
            Изображение PIL

        Returns
        -------
        np.ndarray или None
            Массив признаков или None в случае ошибки

        Examples
        --------
        >>> from PIL import Image
        >>> image = Image.open("image.jpg")
        >>> features = extractor.extract_features(image)
        >>> if features is not None:
        ...     print(f"Размер вектора признаков: {features.shape}")
        """
        try:
            # Применяем трансформации
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Извлекаем признаки
            with torch.no_grad():
                features = self.model(image_tensor)

            # Преобразуем в numpy array и нормализуем
            features = features.cpu().numpy().flatten()
            features = features / np.linalg.norm(features)

            return features

        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return None

    def process_image_batch(self, image_batch: Dict[str, bytes]) -> Dict[str, Dict[str, Any]]:
        """
        Обработка батча изображений {s3_key: image_data}

        Parameters
        ----------
        image_batch : Dict[str, bytes]
            Словарь с данными изображений

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Словарь с результатами обработки

        Examples
        --------
        >>> image_batch = {"image1.jpg": image_data1, "image2.jpg": image_data2}
        >>> results = extractor.process_image_batch(image_batch)
        >>> print(f"Обработано изображений: {len(results)}")
        """
        start_time = time.time()
        batch_results: Dict[str, Dict[str, Any]] = {}

        for s3_key, image_data in image_batch.items():
            # Загружаем изображение из байтов
            image = self.load_image_from_bytes(image_data)
            if image is None:
                self.stats["failed"] += 1
                continue

            # Извлекаем признаки
            features = self.extract_features(image)
            if features is not None:
                batch_results[s3_key] = {"features": features, "s3_key": s3_key}
                self.stats["processed"] += 1
            else:
                self.stats["failed"] += 1

        batch_time = time.time() - start_time
        self.stats["total_time"] += batch_time

        return batch_results

    def get_all_s3_images(self, prefix: str = "") -> List[str]:
        """
        Получение списка всех изображений в S3 bucket

        Parameters
        ----------
        prefix : str, optional
            Префикс для фильтрации файлов (по умолчанию "")

        Returns
        -------
        List[str]
            Список ключей изображений

        Examples
        --------
        >>> image_keys = extractor.get_all_s3_images(prefix="moscow/images/")
        >>> print(f"Найдено изображений: {len(image_keys)}")
        """
        print("Получение списка изображений из S3...")
        image_keys = s3_manager.list_files(prefix=prefix, file_extensions=[".jpg", ".jpeg", ".png", ".webp"])
        print(f"Найдено {len(image_keys)} изображений")
        return image_keys

    def process_all_images(self, batch_size: int = 32) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Пакетная обработка всех изображений из S3

        Parameters
        ----------
        batch_size : int, optional
            Размер батча для обработки (по умолчанию 32)

        Returns
        -------
        Tuple[Dict[str, Dict[str, Any]], List[str]]
            Кортеж из словаря признаков и списка неудачных изображений

        Examples
        --------
        >>> features_dict, failed_images = extractor.process_all_images(batch_size=16)
        >>> print(f"Обработано изображений: {len(features_dict)}")
        >>> print(f"Ошибок: {len(failed_images)}")
        """
        print("Начало обработки всех изображений...")

        # Получаем список всех изображений
        image_keys = self.get_all_s3_images()
        total_images = len(image_keys)

        if total_images == 0:
            print("Не найдено изображений для обработки")
            return {}, []

        features_dict: Dict[str, Dict[str, Any]] = {}
        failed_images: List[str] = []

        # Обработка батчами
        for i in tqdm(range(0, total_images, batch_size)):
            batch_keys = image_keys[i : i + batch_size]

            # Загружаем батч изображений из S3
            print(f"Загрузка батча {i//batch_size + 1}...")
            batch_data = s3_manager.batch_download_bytes(batch_keys)

            # Обрабатываем батч
            batch_results = self.process_image_batch(batch_data)
            features_dict.update(batch_results)

            # Определяем неудачные загрузки
            for key in batch_keys:
                if key not in batch_results and batch_data.get(key) is not None:
                    failed_images.append(key)

            # Очистка памяти CUDA если используется GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Прогресс
            current_processed = len(features_dict)
            print(f"Обработано: {current_processed}/{total_images} " f"({current_processed/total_images*100:.1f}%)")

        # Статистика
        success_rate = self.stats["processed"] / total_images * 100
        avg_time_per_image = self.stats["total_time"] / max(1, self.stats["processed"])

        print(f"\n=== СТАТИСТИКА ОБРАБОТКИ ===")
        print(f"Всего изображений: {total_images}")
        print(f"Успешно обработано: {self.stats['processed']}")
        print(f"Ошибки обработки: {self.stats['failed']}")
        print(f"Успешность: {success_rate:.1f}%")
        print(f"Среднее время на изображение: {avg_time_per_image:.3f} сек")
        print(f"Общее время обработки: {self.stats['total_time']:.2f} сек")

        return features_dict, failed_images

    def get_processing_stats(self) -> Dict[str, int]:
        """
        Получение статистики обработки

        Returns
        -------
        Dict[str, int]
            Словарь со статистикой обработки

        Examples
        --------
        >>> stats = extractor.get_processing_stats()
        >>> print(f"Обработано изображений: {stats['processed']}")
        >>> print(f"Ошибок: {stats['failed']}")
        """
        return self.stats.copy()
