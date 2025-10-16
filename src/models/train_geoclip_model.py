"""
Скрипт для дообучения GeoCLIP модели на данных Москвы и создания FAISS индекса с оптимизацией памяти.
"""

import sys
from pathlib import Path
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Generator, Any
import io
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from PIL import Image
import faiss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import math

# Настраиваем пути проекта
utils_path = Path(__file__).resolve().parent.parent / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

try:
    from path_resolver import setup_project_paths
    setup_project_paths()
except ImportError:
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


class MoscowGeoCLIPDataset(Dataset):
    """Датасет для дообучения GeoCLIP на данных Москвы"""
    
    def __init__(self, image_paths: List[str], coordinates: Dict[str, Tuple[float, float]], 
                 transform=None, text_template: str = "фотография места в Москве с координатами {lat:.4f}, {lon:.4f}"):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transform
        self.text_template = text_template
        
        # Фильтруем только те пути, для которых есть координаты
        self.valid_paths = []
        self.valid_ids = []
        for path in image_paths:
            image_id = os.path.splitext(os.path.basename(path))[0]
            if image_id in coordinates:
                self.valid_paths.append(path)
                self.valid_ids.append(image_id)
        
        logger.info(f"Создан датасет с {len(self.valid_paths)} валидными изображениями")
    
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        image_path = self.valid_paths[idx]
        image_id = self.valid_ids[idx]
        
        try:
            # Загрузка изображения из S3
            image_data = s3_manager.download_bytes(image_path)
            if image_data is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Применяем трансформации
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Создаем пустой тензор правильной формы
                image_tensor = torch.zeros((3, 224, 224))
            
            # Получаем координаты
            lat, lon = self.coordinates[image_id]
            
            # Генерируем текстовое описание
            text = self.text_template.format(lat=lat, lon=lon)
            
            return {
                'image': image_tensor,
                'text': text,
                'coordinates': torch.tensor([lat, lon], dtype=torch.float32),
                'image_path': image_path,
                'image_id': image_id
            }
            
        except Exception as e:
            logger.warning(f"Ошибка при загрузке {image_path}: {e}")
            # Возвращаем пустые данные в случае ошибки
            return {
                'image': torch.zeros((3, 224, 224)),
                'text': "",
                'coordinates': torch.tensor([0.0, 0.0], dtype=torch.float32),
                'image_path': "",
                'image_id': ""
            }


class GeoCLIPFineTuner:
    """Класс для дообучения GeoCLIP модели"""
    
    def __init__(self, model, device, learning_rate=1e-5, margin=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
    def _haversine_distance(self, coords1, coords2):
        """Вычисляет расстояние Хаверсина между координатами в км (работает на GPU)"""
        # coords1: (batch_size, 2) или (1, 2)
        # coords2: (batch_size, 2) или (1, 2)
        
        lat1, lon1 = coords1[:, 0], coords1[:, 1]
        lat2, lon2 = coords2[:, 0], coords2[:, 1]
        
        # Конвертируем в радианы
        lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
        
        # Разницы
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Формула Хаверсина
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distance_km = 6371 * c
        
        return distance_km
    
    def _create_triplets(self, embeddings, coordinates, pos_threshold=0.1, neg_threshold=1.0):
        """Создает триплеты для контрастного обучения (оптимизировано для GPU)"""
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return None, None, None
        
        # Вычисляем матрицу географических расстояний на GPU
        geo_distances = torch.zeros((batch_size, batch_size), device=self.device)
        
        # Векторизованное вычисление расстояний
        for i in range(batch_size):
            # Сравниваем i-ю точку со всеми остальными
            coords_i = coordinates[i].unsqueeze(0)  # (1, 2)
            coords_all = coordinates  # (batch_size, 2)
            
            distances = self._haversine_distance(
                coords_i.expand(batch_size, 2), 
                coords_all
            )
            geo_distances[i] = distances
        
        # Создаем триплеты
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            # Находим positive примеры (близкие географически)
            positive_mask = (geo_distances[i] < pos_threshold) & (geo_distances[i] > 0)
            positive_indices = torch.where(positive_mask)[0]
            
            # Находим negative примеры (далекие географически)
            negative_mask = geo_distances[i] > neg_threshold
            negative_indices = torch.where(negative_mask)[0]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Выбираем случайные positive и negative (используем .item() для получения скаляров)
                pos_idx = positive_indices[torch.randint(0, len(positive_indices), (1,))].item()
                neg_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))].item()
                
                anchors.append(embeddings[i])
                positives.append(embeddings[pos_idx])
                negatives.append(embeddings[neg_idx])
        
        if anchors:
            return (
                torch.stack(anchors),
                torch.stack(positives), 
                torch.stack(negatives)
            )
        return None, None, None
    
    def train_epoch(self, dataloader, epoch):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        num_triplets = 0
        
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch} - Обучение")
        for batch_idx, batch in enumerate(pbar):
            # Пропускаем пустые батчи
            if batch['image'].nelement() == 0 or batch['image'].shape[0] == 0:
                continue
                
            images = batch['image'].to(self.device)
            coordinates = batch['coordinates'].to(self.device)
            
            # Пропускаем батчи с некорректными данными
            if images.shape[0] < 2:  # Нужно минимум 2 изображения для триплетов
                continue
                
            self.optimizer.zero_grad()
            
            # Прямой проход
            embeddings = self.model(images)
            
            # Создаем триплеты и вычисляем потери
            anchors, positives, negatives = self._create_triplets(embeddings, coordinates)
            
            if anchors is not None and anchors.shape[0] > 0:
                # Проверяем размерности
                if anchors.dim() != 2 or positives.dim() != 2 or negatives.dim() != 2:
                    logger.warning(f"Неверные размерности: anchors {anchors.shape}, positives {positives.shape}, negatives {negatives.shape}")
                    continue
                    
                loss = self.criterion(anchors, positives, negatives)
                loss.backward()
                
                # Градиентный clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                batch_triplets = anchors.shape[0]
                total_loss += loss.item() * batch_triplets
                num_triplets += batch_triplets
                num_batches += 1
                
                avg_loss = total_loss / num_triplets if num_triplets > 0 else 0
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{avg_loss:.6f}',
                    'triplets': f'{batch_triplets}'
                })
            
            # Очистка памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_triplets if num_triplets > 0 else 0
        logger.info(f"Эпоха {epoch}: обучение, средний loss: {avg_loss:.6f}, триплетов: {num_triplets}")
        return avg_loss
    
    def validate(self, dataloader, epoch):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        num_triplets = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Эпоха {epoch} - Валидация")
            for batch_idx, batch in enumerate(pbar):
                if batch['image'].nelement() == 0 or batch['image'].shape[0] == 0:
                    continue
                    
                images = batch['image'].to(self.device)
                coordinates = batch['coordinates'].to(self.device)
                
                if images.shape[0] < 2:
                    continue
                
                embeddings = self.model(images)
                anchors, positives, negatives = self._create_triplets(embeddings, coordinates)
                
                if anchors is not None and anchors.shape[0] > 0:
                    # Проверяем размерности
                    if anchors.dim() != 2 or positives.dim() != 2 or negatives.dim() != 2:
                        logger.warning(f"Неверные размерности при валидации: anchors {anchors.shape}, positives {positives.shape}, negatives {negatives.shape}")
                        continue
                        
                    loss = self.criterion(anchors, positives, negatives)
                    batch_triplets = anchors.shape[0]
                    total_loss += loss.item() * batch_triplets
                    num_triplets += batch_triplets
                    num_batches += 1
                    
                    avg_loss = total_loss / num_triplets if num_triplets > 0 else 0
                    pbar.set_postfix({
                        'val_loss': f'{loss.item():.6f}',
                        'avg_val_loss': f'{avg_loss:.6f}',
                        'triplets': f'{batch_triplets}'
                    })
        
        avg_loss = total_loss / num_triplets if num_triplets > 0 else 0
        logger.info(f"Эпоха {epoch}: валидация, средний loss: {avg_loss:.6f}, триплетов: {num_triplets}")
        return avg_loss


def load_coordinates_from_json(json_dir: str) -> Dict[str, Tuple[float, float]]:
    """
    Загружает координаты из JSON файлов.
    """
    coordinates = {}
    
    logger.info(f"Загрузка координат из директории: {json_dir}")
    
    if not os.path.exists(json_dir):
        logger.error(f"Директория с JSON файлами не найдена: {json_dir}")
        return coordinates
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    logger.info(f"Найдено JSON файлов: {len(json_files)}")
    
    for json_file in tqdm(json_files, desc="Парсинг JSON файлов"):
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            items = data if isinstance(data, list) else [data]
                
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


def get_image_paths(s3_prefix: str, sample_size: int = None) -> List[str]:
    """
    Получает пути к изображениям из S3.
    """
    logger.info(f"Загрузка списка изображений из S3 по префиксу: {s3_prefix}")
    
    try:
        image_paths = s3_manager.list_files(prefix=s3_prefix, file_extensions=[".jpg", ".jpeg", ".png"])
        logger.info(f"Найдено изображений в S3: {len(image_paths)}")
        
        # Если указан sample_size, ограничиваем количество
        if sample_size and sample_size < len(image_paths):
            logger.info(f"Ограничиваем выборкой: {sample_size} изображений")
            image_paths = random.sample(image_paths, sample_size)
        
        return image_paths
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке списка изображений из S3: {e}")
        return []


def create_image_transform():
    """Создает трансформации для изображений."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def fine_tune_geoclip(
    image_paths: List[str],
    coordinates: Dict[str, Tuple[float, float]],
    output_model_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    train_ratio: float = 0.8
):
    """
    Дообучение GeoCLIP модели на данных Москвы.
    """
    logger.info("Начало дообучения GeoCLIP модели...")
    
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")
    
    # Загрузка модели
    model = ImageEncoder()
    logger.info(f"Модель загружена, параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Трансформации
    transform = create_image_transform()
    
    # Создание датасета
    dataset = MoscowGeoCLIPDataset(image_paths, coordinates, transform)
    
    if len(dataset) == 0:
        logger.error("Нет данных для обучения")
        return None
    
    # Разделение на train/validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoader'ы с коллайтом для обработки пустых батчей
    def collate_fn(batch):
        # Фильтруем пустые примеры
        batch = [item for item in batch if item['image'].nelement() > 0 and item['image_id']]
        if len(batch) == 0:
            return {
                'image': torch.tensor([]),
                'text': [],
                'coordinates': torch.tensor([]),
                'image_path': [],
                'image_id': []
            }
        
        return {
            'image': torch.stack([item['image'] for item in batch]),
            'text': [item['text'] for item in batch],
            'coordinates': torch.stack([item['coordinates'] for item in batch]),
            'image_path': [item['image_path'] for item in batch],
            'image_id': [item['image_id'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, collate_fn=collate_fn)
    
    logger.info(f"Размер обучающей выборки: {len(train_dataset)}")
    logger.info(f"Размер валидационной выборки: {len(val_dataset)}")
    
    # Инициализация тренера
    trainer = GeoCLIPFineTuner(model, device, learning_rate)
    
    # Обучение
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, epochs + 1):
        logger.info(f"Эпоха {epoch}/{epochs}")
        
        # Обучение
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Валидация
        val_loss = trainer.validate(val_loader, epoch)
        
        # Обновление learning rate (только если был шаг оптимизатора)
        if train_loss > 0:
            trainer.scheduler.step()
        
        # Сохранение истории
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': trainer.scheduler.get_last_lr()[0] if train_loss > 0 else learning_rate
        })
        
        logger.info(f"Эпоха {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'training_history': training_history
            }, output_model_path)
            logger.info(f"Сохранена лучшая модель с val_loss={val_loss:.6f}")
    
    logger.info(f"Дообучение завершено. Лучшая val_loss: {best_val_loss:.6f}")
    
    # Сохранение истории обучения
    history_path = output_model_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return model


class FineTunedGeoCLIPEmbedder:
    """Класс для извлечения эмбеддингов из дообученной модели"""
    
    def __init__(self, model_path: str, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка модели
        self.model = ImageEncoder()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = create_image_transform()
        
        logger.info(f"Загружена дообученная модель из {model_path} на устройство {self.device}")
    
    def extract_embeddings(self, image_paths_batch: List[str], coordinates: Dict[str, Tuple[float, float]], 
                          max_workers: int = 2) -> Tuple[List[np.ndarray], List[str], List[Tuple[float, float]]]:
        """
        Извлекает эмбеддинги для батча изображений.
        """
        embeddings = []
        valid_image_paths = []
        valid_coordinates = []
        
        def process_single_image(image_path: str):
            try:
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                
                # Проверяем наличие координат
                if image_id not in coordinates:
                    return None
                
                # Загружаем изображение
                image_data = s3_manager.download_bytes(image_path)
                if image_data is None:
                    return None
                
                # Обрабатываем изображение
                image = Image.open(io.BytesIO(image_data))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                if image.size[0] < 10 or image.size[1] < 10:
                    return None
                
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Извлекаем эмбеддинг
                with torch.no_grad():
                    embedding = self.model(tensor).cpu().numpy().flatten()
                
                return embedding, image_path, coordinates[image_id]
                
            except Exception as e:
                logger.debug(f"Ошибка при обработке {image_path}: {e}")
                return None
        
        # Обрабатываем изображения параллельно
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_image, path): path 
                for path in image_paths_batch
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Извлечение эмбеддингов"):
                result = future.result()
                if result is not None:
                    embedding, image_path, coords = result
                    embeddings.append(embedding)
                    valid_image_paths.append(image_path)
                    valid_coordinates.append(coords)
        
        return embeddings, valid_image_paths, valid_coordinates


def create_faiss_index_incremental(embedding_generator, output_index_path: str, output_mapping_path: str):
    """
    Создает FAISS индекс инкрементально, обрабатывая данные по частям.
    """
    logger.info("Создание FAISS индекса инкрементально...")
    
    index = None
    mapping_data = []
    total_vectors = 0
    
    # Создаем директории если их нет
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mapping_path), exist_ok=True)
    
    batch_count = 0
    for batch_idx, (embeddings_batch, image_paths_batch, coordinates_batch) in enumerate(embedding_generator):
        if not embeddings_batch:
            logger.info(f"Батч {batch_idx} пустой, пропускаем")
            continue
            
        logger.info(f"Обработка батча {batch_idx + 1} с {len(embeddings_batch)} эмбеддингами")
        
        embeddings_array = np.array(embeddings_batch).astype('float32')
        
        # Создаем индекс при первой итерации
        if index is None:
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)
            logger.info(f"Создан FAISS индекс с размерностью: {dimension}")
        
        # Нормализуем и добавляем в индекс
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        # Сохраняем маппинг
        for i, (image_path, coords) in enumerate(zip(image_paths_batch, coordinates_batch)):
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            mapping_data.append({
                'index_id': total_vectors + i,
                'image_id': image_id,
                's3_key': image_path,
                'latitude': coords[0],
                'longitude': coords[1]
            })
        
        total_vectors += len(embeddings_batch)
        batch_count += 1
        logger.info(f"Обработан батч {batch_idx + 1}, всего векторов: {total_vectors}")
        
        # Очистка памяти
        del embeddings_array, embeddings_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if index is None or total_vectors == 0:
        raise ValueError("Не удалось создать индекс - нет данных")
    
    # Сохраняем индекс и маппинг
    logger.info(f"Сохранение FAISS индекса с {total_vectors} векторами...")
    faiss.write_index(index, output_index_path)
    
    logger.info("Сохранение маппинга...")
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_mapping_path, index=False)
    
    logger.info(f"Индекс и маппинг успешно сохранены. Всего обработано: {total_vectors} изображений")
    
    return index, mapping_df


def main():
    """Основная функция для дообучения GeoCLIP и создания FAISS индекса."""
    
    parser = argparse.ArgumentParser(description='Дообучение GeoCLIP и создание FAISS индекса')
    parser.add_argument('--fine-tune', action='store_true', default=True,
                       help='Выполнить дообучение модели')
    parser.add_argument('--fine-tune-sample-size', type=int, default=10000,
                       help='Количество изображений для дообучения')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Количество эпох дообучения')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Размер батча для дообучения')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate для дообучения')
    parser.add_argument('--sample-size', type=int, default=500000, 
                       help='Количество изображений для создания индекса')
    parser.add_argument('--index-batch-size', type=int, default=5000,
                       help='Размер батча для создания индекса')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Максимальное количество потоков')
    parser.add_argument('--debug', action='store_true', 
                       help='Включить debug режим')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Пути к данным
    S3_IMAGE_PREFIX = "site/raw_data/"
    JSON_COORDINATES_DIR = "logs/download_data/cache_moscow_images/data_json/"
    OUTPUT_MODEL_PATH = "models/fine_tuned_geoclip_moscow.pth"
    OUTPUT_INDEX_PATH = "data/index/faiss_index_fine_tuned.bin"
    OUTPUT_MAPPING_PATH = "data/index/image_mapping_fine_tuned.csv"
    
    logger.info("Начало процесса дообучения GeoCLIP и создания FAISS индекса")
    
    try:
        # 1. Загружаем координаты
        coordinates = load_coordinates_from_json(JSON_COORDINATES_DIR)
        if not coordinates:
            logger.error("Не удалось загрузить координаты")
            return
        
        # 2. Получаем пути к изображениям
        image_paths = get_image_paths(S3_IMAGE_PREFIX, args.sample_size)
        if not image_paths:
            logger.error("Не удалось загрузить пути к изображениям")
            return
        
        # 3. Дообучение модели (если включено)
        if args.fine_tune:
            logger.info("Запуск дообучения модели...")
            fine_tune_paths = image_paths[:args.fine_tune_sample_size]
            
            fine_tuned_model = fine_tune_geoclip(
                image_paths=fine_tune_paths,
                coordinates=coordinates,
                output_model_path=OUTPUT_MODEL_PATH,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            if fine_tuned_model is None:
                logger.error("Дообучение не удалось")
                return
        else:
            logger.info("Пропуск дообучения, использование предобученной модели")
            OUTPUT_MODEL_PATH = None  # Будем использовать стандартную модель
        
        # 4. Создаем эмбеддер
        if args.fine_tune and os.path.exists(OUTPUT_MODEL_PATH):
            embedder = FineTunedGeoCLIPEmbedder(OUTPUT_MODEL_PATH)
            logger.info("Используется дообученная модель")
        else:
            # Используем стандартную модель
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ImageEncoder().to(device).eval()
            embedder = type('MockEmbedder', (), {
                'extract_embeddings': lambda self, paths, coords, workers: encode_images_streaming(
                    paths, coords, model, 
                    create_image_transform(), device, workers
                ),
                'device': device
            })()
            logger.info("Используется стандартная модель GeoCLIP")
        
        # 5. Создаем генератор для батчей изображений
        def embedding_generator():
            total_processed = 0
            for i in range(0, len(image_paths), args.index_batch_size):
                batch_paths = image_paths[i:i + args.index_batch_size]
                if args.sample_size and total_processed >= args.sample_size:
                    break
                    
                logger.info(f"Обработка батча из {len(batch_paths)} изображений")
                embeddings, valid_paths, valid_coords = embedder.extract_embeddings(
                    batch_paths, coordinates, args.max_workers
                )
                
                if embeddings:
                    logger.info(f"Успешно обработано {len(embeddings)} изображений в батче")
                    yield embeddings, valid_paths, valid_coords
                    total_processed += len(embeddings)
                else:
                    logger.warning("Батч не содержит валидных эмбеддингов")
        
        # 6. Создаем индекс инкрементально
        create_faiss_index_incremental(
            embedding_generator(), 
            OUTPUT_INDEX_PATH, 
            OUTPUT_MAPPING_PATH
        )
        
        logger.info("Процесс успешно завершен!")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# Добавляем функцию encode_images_streaming из оригинального кода для совместимости
def encode_images_streaming(
    image_paths_batch: List[str],
    coordinates: Dict[str, Tuple[float, float]],
    image_encoder,
    transform,
    device,
    max_workers: int = 2
) -> Tuple[List[np.ndarray], List[str], List[Tuple[float, float]]]:
    """
    Кодирование батча изображений с потоковой обработкой.
    """
    embeddings = []
    valid_image_paths = []
    valid_coordinates = []
    
    def process_single_image(image_path: str):
        try:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # Проверяем наличие координат
            if image_id not in coordinates:
                return None
                
            # Загружаем изображение
            image_data = s3_manager.download_bytes(image_path)
            if image_data is None:
                return None
                
            # Обрабатываем изображение
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            if image.size[0] < 10 or image.size[1] < 10:
                return None
                
            tensor = transform(image).unsqueeze(0).to(device)
            
            # Кодируем изображение
            with torch.no_grad():
                embedding = image_encoder(tensor).cpu().numpy().flatten()
            
            return embedding, image_path, coordinates[image_id]
            
        except Exception as e:
            logger.debug(f"Ошибка при обработке {image_path}: {e}")
            return None
    
    # Обрабатываем изображения параллельно
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, path): path 
            for path in image_paths_batch
        }
        
        processed_results = []
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка изображений"):
            result = future.result()
            if result is not None:
                processed_results.append(result)
    
    # Разделяем результаты
    if processed_results:
        embeddings, valid_image_paths, valid_coordinates = zip(*processed_results)
        embeddings = list(embeddings)
        valid_image_paths = list(valid_image_paths)
        valid_coordinates = list(valid_coordinates)
    
    # Очистка памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return embeddings, valid_image_paths, valid_coordinates


if __name__ == "__main__":
    main()