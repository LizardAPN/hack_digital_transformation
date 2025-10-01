"""
Скрипт для обучения GeoCLIP модели и создания FAISS индекса на основе изображений из S3 и координат из JSON файлов.
"""

import sys
from pathlib import Path
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
import io
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import faiss
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

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

from geoclip import ImageEncoder
from src.utils.config import s3_manager, DATA_PATHS

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_coordinates_from_json(json_dir: str) -> Dict[str, Tuple[float, float]]:
    """
    Загружает координаты из JSON файлов.
    
    Args:
        json_dir: Путь к директории с JSON файлами
        
    Returns:
        Словарь {image_id: (lat, lon)}
    """
    coordinates = {}
    
    logger.info(f"Загрузка координат из директории: {json_dir}")
    
    # Проверяем существование директории
    if not os.path.exists(json_dir):
        logger.error(f"Директория с JSON файлами не найдена: {json_dir}")
        return coordinates
    
    # Проходим по всем JSON файлам в директории
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    logger.info(f"Найдено JSON файлов: {len(json_files)}")
    
    for json_file in tqdm(json_files, desc="Парсинг JSON файлов"):
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Обрабатываем каждый элемент в JSON файле
            if isinstance(data, list):
                items = data
            else:
                items = [data]
                
            for item in items:
                if 'id' in item and 'geometry' in item:
                    image_id = item['id']
                    if 'coordinates' in item['geometry']:
                        coords = item['geometry']['coordinates']
                        if len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            coordinates[image_id] = (lat, lon)
                        
        except Exception as e:
            logger.warning(f"Ошибка при обработке файла {json_file}: {e}")
            continue
    
    logger.info(f"Загружено координат для {len(coordinates)} изображений")
    return coordinates


def load_images_from_s3(s3_prefix: str, sample_size: int = None) -> List[str]:
    """
    Загружает список путей к изображениям из S3.
    
    Args:
        s3_prefix: Префикс пути в S3
        sample_size: Количество изображений для выборки (None для всех)
        
    Returns:
        Список путей к изображениям
    """
    logger.info(f"Загрузка списка изображений из S3 по префиксу: {s3_prefix}")
    
    try:
        # Получаем список объектов в S3 по префиксу
        image_paths = s3_manager.list_files(prefix=s3_prefix, file_extensions=[".jpg", ".jpeg", ".png"])
        logger.info(f"Найдено изображений в S3: {len(image_paths)}")
        
        # Если указан размер выборки, берем только часть изображений
        if sample_size and sample_size < len(image_paths):
            logger.info(f"Выбираем случайную выборку из {sample_size} изображений")
            import random
            random.shuffle(image_paths)
            image_paths = image_paths[:sample_size]
            logger.info(f"Выбрано {len(image_paths)} изображений для обработки")
        
        return image_paths
    except Exception as e:
        logger.error(f"Ошибка при загрузке списка изображений из S3: {e}")
        return []


def extract_image_id_from_path(image_path: str) -> str:
    """
    Извлекает ID изображения из пути.
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        ID изображения
    """
    # Предполагаем, что ID - это имя файла без расширения
    filename = os.path.basename(image_path)
    image_id = os.path.splitext(filename)[0]
    return image_id


def create_image_transform():
    """
    Создает трансформации для изображений, совместимые с GeoCLIP.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_and_validate_image(image_data: bytes, transform) -> Optional[torch.Tensor]:
    """
    Предобрабатывает и валидирует изображение.
    
    Args:
        image_data: Данные изображения в байтах
        transform: Трансформации для изображения
        
    Returns:
        Тензор изображения или None при ошибке
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Базовые проверки размера изображения
        if image.size[0] < 10 or image.size[1] < 10:
            logger.warning("Изображение слишком маленькое")
            return None
        
        # Применяем трансформации
        tensor = transform(image)
        return tensor
        
    except Exception as e:
        logger.warning(f"Ошибка при открытии изображения: {e}")
        return None


def encode_images_simple(image_paths: List[str], coordinates: Dict[str, Tuple[float, float]], 
                        batch_size: int = 32) -> Tuple[List[np.ndarray], List[str], List[Tuple[float, float]]]:
    """
    Упрощенная версия кодирования изображений.
    
    Args:
        image_paths: Список путей к изображениям
        coordinates: Словарь координат {image_id: (lat, lon)}
        batch_size: Размер батча для обработки
        
    Returns:
        Кортеж (эмбеддинги, пути к изображениям, координаты)
    """
    logger.info("Инициализация GeoCLIP модели...")
    
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")
    
    # Инициализируем модель
    image_encoder = ImageEncoder().to(device).eval()
    
    # Создаем трансформации
    transform = create_image_transform()
    
    embeddings = []
    valid_image_paths = []
    valid_coordinates = []
    
    logger.info(f"Кодирование {len(image_paths)} изображений...")
    
    # Сначала соберем все валидные изображения
    valid_images = []
    valid_data = []
    
    for image_path in tqdm(image_paths, desc="Подготовка изображений"):
        try:
            # Извлекаем ID изображения
            image_id = extract_image_id_from_path(image_path)
            
            # Проверяем, есть ли координаты для этого изображения
            if image_id not in coordinates:
                logger.debug(f"Нет координат для изображения: {image_path}")
                continue
            
            # Загружаем изображение из S3
            image_data = s3_manager.download_bytes(image_path)
            if image_data is None:
                logger.warning(f"Не удалось загрузить изображение: {image_path}")
                continue
            
            # Предобрабатываем изображение
            tensor = preprocess_and_validate_image(image_data, transform)
            if tensor is None:
                logger.warning(f"Не удалось обработать изображение: {image_path}")
                continue
            
            valid_images.append(tensor)
            valid_data.append((image_path, image_id))
            
        except Exception as e:
            logger.warning(f"Ошибка при обработке изображения {image_path}: {e}")
            continue
    
    logger.info(f"Подготовлено {len(valid_images)} валидных изображений для кодирования")
    
    # Теперь кодируем батчами
    for i in tqdm(range(0, len(valid_images), batch_size), desc="Кодирование изображений"):
        batch_tensors = valid_images[i:i+batch_size]
        batch_data = valid_data[i:i+batch_size]
        
        if not batch_tensors:
            continue
            
        try:
            # Создаем батч
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            # Получаем эмбеддинги
            with torch.no_grad():
                batch_embeddings = image_encoder(batch_tensor)
                batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Сохраняем результаты
            for j, (image_path, image_id) in enumerate(batch_data):
                embedding = batch_embeddings[j]
                
                # Проверяем размерность эмбеддинга
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                embeddings.append(embedding)
                valid_image_paths.append(image_path)
                valid_coordinates.append(coordinates[image_id])
                
        except Exception as e:
            logger.error(f"Ошибка при кодировании батча: {e}")
            # Продолжаем обработку следующих батчей
            continue
    
    logger.info(f"Успешно закодировано {len(embeddings)} изображений")
    return embeddings, valid_image_paths, valid_coordinates


def create_faiss_index(embeddings: List[np.ndarray], image_paths: List[str]) -> Tuple[faiss.Index, pd.DataFrame]:
    """
    Создает FAISS индекс и маппинг.
    """
    logger.info("Создание FAISS индекса...")
    
    if not embeddings:
        raise ValueError("Нет эмбеддингов для создания индекса")
    
    # Конвертируем в numpy массив
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Проверяем размерность
    if embeddings_array.ndim != 2:
        logger.error(f"Некорректная размерность эмбеддингов: {embeddings_array.shape}")
        raise ValueError(f"Ожидалась 2D матрица, получена {embeddings_array.ndim}D")
    
    logger.info(f"Размерность эмбеддингов: {embeddings_array.shape}")
    
    # Нормализуем для косинусного расстояния
    faiss.normalize_L2(embeddings_array)
    
    # Создаем FAISS индекс
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    logger.info(f"FAISS индекс создан. Размерность: {dimension}, количество векторов: {index.ntotal}")
    
    # Создаем маппинг - ИСПРАВЛЕННАЯ ЧАСТЬ
    mapping_data = []
    for i, image_path in enumerate(image_paths):
        # Извлекаем ID из полного пути
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        
        mapping_data.append({
            'index_id': i,
            'image_id': image_id,  # Только ID (например "14141")
            's3_key': image_path   # Полный путь в S3 (например "site/raw_data/14141.jpeg")
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    
    return index, mapping_df


def save_index_and_mapping(index: faiss.Index, mapping_df: pd.DataFrame, 
                          index_path: str, mapping_path: str):
    """
    Сохраняет FAISS индекс и маппинг.
    
    Args:
        index: FAISS индекс
        mapping_df: DataFrame с маппингом
        index_path: Путь для сохранения индекса
        mapping_path: Путь для сохранения маппинга
    """
    logger.info(f"Сохранение FAISS индекса в: {index_path}")
    logger.info(f"Сохранение маппинга в: {mapping_path}")
    
    # Создаем директории если их нет
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    
    # Сохраняем индекс
    faiss.write_index(index, index_path)
    
    # Сохраняем маппинг
    mapping_df.to_csv(mapping_path, index=False)
    
    logger.info("Индекс и маппинг успешно сохранены")


def main():
    """Основная функция для обучения GeoCLIP и создания FAISS индекса."""
    
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='Обучение GeoCLIP и создание FAISS индекса')
    parser.add_argument('--sample-size', type=int, default=1000, 
                       help='Количество изображений для обработки (по умолчанию 1000)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Размер батча для обработки (по умолчанию 32)')
    parser.add_argument('--debug', action='store_true', 
                       help='Включить debug режим с более подробным логированием')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Пути к данным
    S3_IMAGE_PREFIX = "site/raw_data/"  # Префикс для изображений в S3
    JSON_COORDINATES_DIR = "/home/lizardapn/Hack_digital/hack_digital_transformation/logs/download_data/cache_moscow_images"  # Директория с JSON файлами
    OUTPUT_INDEX_PATH = "data/index/faiss_index.bin"  # Путь для сохранения индекса
    OUTPUT_MAPPING_PATH = "data/index/image_mapping.csv"  # Путь для сохранения маппинга
    
    logger.info("Начало процесса обучения GeoCLIP и создания FAISS индекса")
    if args.sample_size:
        logger.info(f"Используется выборка из {args.sample_size} изображений")
    logger.info(f"Размер батча: {args.batch_size}")
    
    # 1. Загружаем координаты из JSON файлов
    coordinates = load_coordinates_from_json(JSON_COORDINATES_DIR)
    if not coordinates:
        logger.error("Не удалось загрузить координаты из JSON файлов")
        return
    
    # 2. Загружаем список изображений из S3
    image_paths = load_images_from_s3(S3_IMAGE_PREFIX, args.sample_size)
    if not image_paths:
        logger.error("Не удалось загрузить список изображений из S3")
        return
    
    # 3. Кодируем изображения в эмбеддинги
    embeddings, valid_image_paths, valid_coordinates = encode_images_simple(image_paths, coordinates, args.batch_size)
    if not embeddings:
        logger.error("Не удалось закодировать изображения")
        return
    
    # 4. Создаем FAISS индекс
    try:
        index, mapping_df = create_faiss_index(embeddings, valid_image_paths)
    except Exception as e:
        logger.error(f"Ошибка при создании FAISS индекса: {e}")
        return
    
    # 5. Сохраняем индекс и маппинг
    save_index_and_mapping(index, mapping_df, OUTPUT_INDEX_PATH, OUTPUT_MAPPING_PATH)
    
    logger.info("Процесс успешно завершен!")


if __name__ == "__main__":
    main()