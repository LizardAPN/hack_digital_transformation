"""
Оптимизированный скрипт для дообучения GeoCLIP модели на данных Москвы с использованием нескольких GPU.
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
import shutil
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

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
    """Оптимизированный датасет для дообучения GeoCLIP на данных Москвы"""
    
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


class OptimizedGeoCLIPFineTuner:
    """Оптимизированный класс для дообучения GeoCLIP модели с поддержкой многокарточности"""
    
    def __init__(self, model, device, learning_rate=1e-5, margin=1.0, use_amp=True):
        # Используем DataParallel если доступно несколько GPU
        if torch.cuda.device_count() > 1:
            logger.info(f"Используется {torch.cuda.device_count()} GPU")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
            
        self.model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Оптимизированный оптимизатор
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
    def _haversine_distance(self, coords1, coords2):
        """Векторизованное вычисление расстояния Хаверсина"""
        lat1, lon1 = coords1[:, 0], coords1[:, 1]
        lat2, lon2 = coords2[:, 0], coords2[:, 1]
        
        # Конвертируем в радианы
        lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
        
        # Разницы
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Векторизованная формула Хаверсина
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distance_km = 6371 * c
        
        return distance_km
    
    def _create_triplets_batch(self, embeddings, coordinates, pos_threshold=0.5, neg_threshold=2.0):
        """Векторизованное создание триплетов для всего батча"""
        batch_size = embeddings.size(0)
        if batch_size < 3:  # Нужно минимум 3 изображения для триплетов
            return None, None, None
        
        # Вычисляем матрицу географических расстояний на GPU
        coords_expanded1 = coordinates.unsqueeze(1).expand(batch_size, batch_size, 2)
        coords_expanded2 = coordinates.unsqueeze(0).expand(batch_size, batch_size, 2)
        
        # Векторизованное вычисление расстояний
        lat1, lon1 = coords_expanded1[..., 0], coords_expanded1[..., 1]
        lat2, lon2 = coords_expanded2[..., 0], coords_expanded2[..., 1]
        
        lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        geo_distances = 6371 * c
        
        # Создаем маски для positive и negative примеров
        eye_mask = torch.eye(batch_size, device=self.device).bool()
        geo_distances_masked = geo_distances.clone()
        geo_distances_masked[eye_mask] = float('inf')  # Исключаем диагональ
        
        positive_mask = (geo_distances_masked < pos_threshold) & (geo_distances_masked > 0)
        negative_mask = geo_distances_masked > neg_threshold
        
        # Создаем триплеты
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            positive_indices = torch.where(positive_mask[i])[0]
            negative_indices = torch.where(negative_mask[i])[0]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Берем несколько триплетов для каждого anchor
                for _ in range(min(2, len(positive_indices))):  # Ограничиваем количество триплетов на anchor
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
        """Оптимизированный эпох обучения с mixed precision"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        num_triplets = 0
        
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch} - Обучение")
        for batch_idx, batch in enumerate(pbar):
            # Пропускаем пустые батчи
            if batch['image'].nelement() == 0 or batch['image'].shape[0] < 3:
                continue
                
            images = batch['image'].to(self.device, non_blocking=True)
            coordinates = batch['coordinates'].to(self.device, non_blocking=True)
                
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                embeddings = self.model(images)
                anchors, positives, negatives = self._create_triplets_batch(embeddings, coordinates)
            
            if anchors is not None and anchors.shape[0] > 0:
                # Mixed precision backward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self.criterion(anchors, positives, negatives)
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Градиентный clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
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
        
        avg_loss = total_loss / num_triplets if num_triplets > 0 else 0
        logger.info(f"Эпоха {epoch}: обучение, средний loss: {avg_loss:.6f}, триплетов: {num_triplets}")
        return avg_loss
    
    def validate(self, dataloader, epoch):
        """Оптимизированная валидация"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        num_triplets = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Эпоха {epoch} - Валидация")
            for batch_idx, batch in enumerate(pbar):
                if batch['image'].nelement() == 0 or batch['image'].shape[0] < 3:
                    continue
                    
                images = batch['image'].to(self.device, non_blocking=True)
                coordinates = batch['coordinates'].to(self.device, non_blocking=True)
                
                # Mixed precision для валидации
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    embeddings = self.model(images)
                    anchors, positives, negatives = self._create_triplets_batch(embeddings, coordinates)
                
                if anchors is not None and anchors.shape[0] > 0:
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


def safe_save_model(checkpoint, filepath, max_retries=3):
    """Безопасное сохранение модели с повторными попытками"""
    for attempt in range(max_retries):
        try:
            temp_path = filepath + f".temp_{attempt}"
            torch.save(checkpoint, temp_path)
            shutil.move(temp_path, filepath)
            logger.info(f"Модель успешно сохранена: {filepath}")
            return True
        except Exception as e:
            logger.warning(f"Попытка {attempt + 1} сохранения не удалась: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if attempt == max_retries - 1:
                logger.error(f"Не удалось сохранить модель после {max_retries} попыток")
                return False
    return False


def haversine_distance(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние Хаверсина между двумя точками в км"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return 6371 * c


def evaluate_retrieval_metrics(embedder, test_image_paths, coordinates, 
                              distance_thresholds=[0.1, 0.5, 1.0, 2.0, 5.0], 
                              k_values=[1, 5, 10], sample_size=1000):
    """Оценивает метрики retrieval (recall, precision) на разных расстояниях"""
    logger.info("Вычисление метрик retrieval...")
    
    if len(test_image_paths) > sample_size:
        test_sample = random.sample(test_image_paths, sample_size)
    else:
        test_sample = test_image_paths
    
    logger.info(f"Тестирование на {len(test_sample)} изображениях")
    
    embeddings, valid_paths, valid_coords = embedder.extract_embeddings(
        test_sample, coordinates, max_workers=4
    )
    
    if not embeddings:
        logger.error("Не удалось извлечь эмбеддинги для тестирования")
        return {}
    
    embeddings_array = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_array)
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    metrics = {}
    
    for k in k_values:
        distances, indices = index.search(embeddings_array, k + 1)
        
        recall_at_k = {f"recall@{k}_{thresh}km": [] for thresh in distance_thresholds}
        precision_at_k = {f"precision@{k}_{thresh}km": [] for thresh in distance_thresholds}
        
        for i in range(len(embeddings)):
            query_coords = valid_coords[i]
            
            for threshold in distance_thresholds:
                relevant_count = 0
                
                for j in range(1, k + 1):
                    if indices[i][j] < len(valid_coords):
                        neighbor_coords = valid_coords[indices[i][j]]
                        distance = haversine_distance(
                            query_coords[0], query_coords[1],
                            neighbor_coords[0], neighbor_coords[1]
                        )
                        
                        if distance <= threshold:
                            relevant_count += 1
                
                recall = 1.0 if relevant_count > 0 else 0.0
                recall_at_k[f"recall@{k}_{threshold}km"].append(recall)
                precision = relevant_count / k
                precision_at_k[f"precision@{k}_{threshold}km"].append(precision)
        
        for threshold in distance_thresholds:
            metrics[f"recall@{k}_{threshold}km"] = np.mean(recall_at_k[f"recall@{k}_{threshold}km"])
            metrics[f"precision@{k}_{threshold}km"] = np.mean(precision_at_k[f"precision@{k}_{threshold}km"])
    
    for threshold in distance_thresholds:
        recalls = [metrics[f"recall@{k}_{threshold}km"] for k in k_values if f"recall@{k}_{threshold}km" in metrics]
        precisions = [metrics[f"precision@{k}_{threshold}km"] for k in k_values if f"precision@{k}_{threshold}km" in metrics]
        
        if recalls:
            metrics[f"mean_recall_{threshold}km"] = np.mean(recalls)
        if precisions:
            metrics[f"mean_precision_{threshold}km"] = np.mean(precisions)
    
    logger.info("Метрики retrieval:")
    for threshold in distance_thresholds:
        logger.info(f"Порог {threshold} км:")
        for k in k_values:
            recall_key = f"recall@{k}_{threshold}km"
            precision_key = f"precision@{k}_{threshold}km"
            if recall_key in metrics:
                logger.info(f"  Recall@{k}: {metrics[recall_key]:.4f}")
            if precision_key in metrics:
                logger.info(f"  Precision@{k}: {metrics[precision_key]:.4f}")
    
    return metrics


def load_coordinates_from_json(json_dir: str) -> Dict[str, Tuple[float, float]]:
    """Загружает координаты из JSON файлов"""
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
    """Получает пути к изображениям из S3"""
    logger.info(f"Загрузка списка изображений из S3 по префиксу: {s3_prefix}")
    
    try:
        image_paths = s3_manager.list_files(prefix=s3_prefix, file_extensions=[".jpg", ".jpeg", ".png"])
        logger.info(f"Найдено изображений в S3: {len(image_paths)}")
        
        if sample_size and sample_size < len(image_paths):
            logger.info(f"Ограничиваем выборкой: {sample_size} изображений")
            image_paths = random.sample(image_paths, sample_size)
        
        return image_paths
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке списка изображений из S3: {e}")
        return []


def create_image_transform():
    """Создает трансформации для изображений"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def fine_tune_geoclip_optimized(
    image_paths: List[str],
    coordinates: Dict[str, Tuple[float, float]],
    output_model_path: str,
    epochs: int = 10,
    batch_size: int = 64,  # Увеличиваем батч для H100
    learning_rate: float = 1e-5,
    train_ratio: float = 0.8,
    compute_metrics: bool = False,
    use_amp: bool = True
):
    """Оптимизированное дообучение GeoCLIP модели"""
    logger.info("Начало оптимизированного дообучения GeoCLIP модели...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")
    
    model = ImageEncoder()
    logger.info(f"Модель загружена, параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    transform = create_image_transform()
    dataset = MoscowGeoCLIPDataset(image_paths, coordinates, transform)
    
    if len(dataset) == 0:
        logger.error("Нет данных для обучения")
        return None
    
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    def collate_fn(batch):
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
    
    # Увеличиваем num_workers для быстрой загрузки
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    logger.info(f"Размер обучающей выборки: {len(train_dataset)}")
    logger.info(f"Размер валидационной выборки: {len(val_dataset)}")
    
    trainer = OptimizedGeoCLIPFineTuner(model, device, learning_rate, use_amp=use_amp)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, epochs + 1):
        logger.info(f"Эпоха {epoch}/{epochs}")
        
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)
        
        if train_loss > 0:
            trainer.scheduler.step()
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': trainer.scheduler.get_last_lr()[0] if train_loss > 0 else learning_rate
        })
        
        logger.info(f"Эпоха {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            
            # Сохраняем оригинальную модель (без DataParallel wrapper)
            model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'training_history': training_history
            }
            
            if safe_save_model(checkpoint, output_model_path):
                logger.info(f"Сохранена лучшая модель с val_loss={val_loss:.6f}")
    
    logger.info(f"Дообучение завершено. Лучшая val_loss: {best_val_loss:.6f}")
    
    history_path = output_model_path.replace('.pth', '_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"История обучения сохранена: {history_path}")
    except Exception as e:
        logger.error(f"Не удалось сохранить историю обучения: {e}")
    
    return model


class OptimizedFineTunedGeoCLIPEmbedder:
    """Оптимизированный класс для извлечения эмбеддингов"""
    
    def __init__(self, model_path: str, device=None):
        if device is None:
            try:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    _ = torch.tensor([1.0]).cuda()
                else:
                    device = torch.device("cpu")
            except Exception:
                device = torch.device("cpu")
                logger.warning("CUDA недоступна, используется CPU")
        
        self.device = device
        self.model = ImageEncoder()
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            # Используем DataParallel для инференса если доступно несколько GPU
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = create_image_transform()
            
            logger.info(f"Загружена дообученная модель из {model_path} на устройство {self.device}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def extract_embeddings(self, image_paths_batch: List[str], coordinates: Dict[str, Tuple[float, float]], 
                          max_workers: int = 4) -> Tuple[List[np.ndarray], List[str], List[Tuple[float, float]]]:
        """Оптимизированное извлечение эмбеддингов"""
        embeddings = []
        valid_image_paths = []
        valid_coordinates = []
        
        def process_single_image(image_path: str):
            try:
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                
                if image_id not in coordinates:
                    return None
                
                image_data = s3_manager.download_bytes(image_path)
                if image_data is None:
                    return None
                
                image = Image.open(io.BytesIO(image_data))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                if image.size[0] < 10 or image.size[1] < 10:
                    return None
                
                tensor = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    embedding = self.model(tensor).cpu().numpy().flatten()
                
                return embedding, image_path, coordinates[image_id]
                
            except Exception as e:
                logger.debug(f"Ошибка при обработке {image_path}: {e}")
                return None
        
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


def main():
    """Основная функция с оптимизированными параметрами для H100"""
    
    parser = argparse.ArgumentParser(description='Оптимизированное дообучение GeoCLIP')
    parser.add_argument('--fine-tune', action='store_true', default=True,
                       help='Выполнить дообучение модели')
    parser.add_argument('--fine-tune-sample-size', type=int, default=20000,
                       help='Количество изображений для дообучения')
    parser.add_argument('--epochs', type=int, default=5,  # Уменьшаем эпохи, увеличиваем батч
                       help='Количество эпох дообучения')
    parser.add_argument('--batch-size', type=int, default=128,  # Большой батч для H100
                       help='Размер батча для дообучения')
    parser.add_argument('--learning-rate', type=float, default=2e-5,  # Увеличиваем LR для большого батча
                       help='Learning rate для дообучения')
    parser.add_argument('--sample-size', type=int, default=500000, 
                       help='Количество изображений для создания индекса')
    parser.add_argument('--index-batch-size', type=int, default=10000,  # Увеличиваем батч для индекса
                       help='Размер батча для создания индекса')
    parser.add_argument('--max-workers', type=int, default=8,  # Больше воркеров для H100
                       help='Максимальное количество потоков')
    parser.add_argument('--use-cpu', action='store_true',
                       help='Принудительно использовать CPU')
    parser.add_argument('--compute-metrics', action='store_true', default=True,
                       help='Вычислять метрики recall и precision')
    parser.add_argument('--no-amp', action='store_true',
                       help='Отключить mixed precision')
    parser.add_argument('--debug', action='store_true', 
                       help='Включить debug режим')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Пути к данным
    S3_IMAGE_PREFIX = "site/raw_data/"
    JSON_COORDINATES_DIR = "logs/download_data/cache_moscow_images/data_json/"
    OUTPUT_MODEL_PATH = "models/fine_tuned_geoclip_moscow_optimized.pth"
    OUTPUT_INDEX_PATH = "data/index/faiss_index_fine_tuned_optimized.bin"
    OUTPUT_MAPPING_PATH = "data/index/image_mapping_fine_tuned_optimized.csv"
    
    logger.info("Начало оптимизированного процесса дообучения GeoCLIP")
    
    try:
        coordinates = load_coordinates_from_json(JSON_COORDINATES_DIR)
        if not coordinates:
            logger.error("Не удалось загрузить координаты")
            return
        
        image_paths = get_image_paths(S3_IMAGE_PREFIX, args.sample_size)
        if not image_paths:
            logger.error("Не удалось загрузить пути к изображениям")
            return
        
        if args.use_cpu:
            device = torch.device("cpu")
            logger.info("Принудительно используется CPU")
        else:
            try:
                device = torch.device("cuda")
                logger.info(f"Используется GPU: {torch.cuda.get_device_name()}")
                logger.info(f"Доступно GPU: {torch.cuda.device_count()}")
            except Exception as e:
                device = torch.device("cpu")
                logger.warning(f"CUDA недоступна, используется CPU. Ошибка: {e}")
        
        if args.fine_tune:
            logger.info("Запуск оптимизированного дообучения модели...")
            fine_tune_paths = image_paths[:args.fine_tune_sample_size]
            
            fine_tuned_model = fine_tune_geoclip_optimized(
                image_paths=fine_tune_paths,
                coordinates=coordinates,
                output_model_path=OUTPUT_MODEL_PATH,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                compute_metrics=args.compute_metrics,
                use_amp=not args.no_amp
            )
            
            if fine_tuned_model is None:
                logger.error("Дообучение не удалось")
                return
        else:
            logger.info("Пропуск дообучения, использование предобученной модели")
            OUTPUT_MODEL_PATH = None
        
        if args.fine_tune and os.path.exists(OUTPUT_MODEL_PATH):
            embedder = OptimizedFineTunedGeoCLIPEmbedder(OUTPUT_MODEL_PATH, device)
            logger.info("Используется дообученная модель")
        else:
            model = ImageEncoder()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device).eval()
            embedder = type('MockEmbedder', (), {
                'extract_embeddings': lambda self, paths, coords, workers: encode_images_streaming(
                    paths, coords, model, 
                    create_image_transform(), device, workers
                ),
                'device': device
            })()
            logger.info("Используется стандартная модель GeoCLIP")

        if args.compute_metrics:
            logger.info("Вычисление метрик для текущей модели...")
            try:
                test_sample_size = min(2000, len(image_paths))  # Увеличиваем выборку для метрик
                test_paths = random.sample(image_paths, test_sample_size)
                
                metrics = evaluate_retrieval_metrics(
                    embedder, test_paths, coordinates,
                    distance_thresholds=[0.1, 0.5, 1.0, 2.0, 5.0],
                    k_values=[1, 5, 10],
                    sample_size=test_sample_size
                )
                
                metrics_model_type = "fine_tuned_optimized" if args.fine_tune else "standard"
                metrics_path = f"models/{metrics_model_type}_geoclip_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Метрики {metrics_model_type} модели сохранены: {metrics_path}")
                
            except Exception as e:
                logger.error(f"Ошибка при вычислении метрик: {e}")
        
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
        
        create_faiss_index_incremental(
            embedding_generator(), 
            OUTPUT_INDEX_PATH, 
            OUTPUT_MAPPING_PATH
        )
        
        logger.info("Оптимизированный процесс успешно завершен!")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# Остальные функции остаются без изменений
def create_faiss_index_incremental(embedding_generator, output_index_path: str, output_mapping_path: str):
    """Создает FAISS индекс инкрементально"""
    logger.info("Создание FAISS индекса инкрементально...")
    
    index = None
    mapping_data = []
    total_vectors = 0
    
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mapping_path), exist_ok=True)
    
    for batch_idx, (embeddings_batch, image_paths_batch, coordinates_batch) in enumerate(embedding_generator):
        if not embeddings_batch:
            continue
            
        logger.info(f"Обработка батча {batch_idx + 1} с {len(embeddings_batch)} эмбеддингами")
        
        embeddings_array = np.array(embeddings_batch).astype('float32')
        
        if index is None:
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)
            logger.info(f"Создан FAISS индекс с размерностью: {dimension}")
        
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
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
        logger.info(f"Обработан батч {batch_idx + 1}, всего векторов: {total_vectors}")
        
        del embeddings_array, embeddings_batch
        safe_cuda_cleanup()
    
    if index is None or total_vectors == 0:
        raise ValueError("Не удалось создать индекс - нет данных")
    
    logger.info(f"Сохранение FAISS индекса с {total_vectors} векторами...")
    faiss.write_index(index, output_index_path)
    
    logger.info("Сохранение маппинга...")
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_mapping_path, index=False)
    
    logger.info(f"Индекс и маппинг успешно сохранены. Всего обработано: {total_vectors} изображений")
    
    return index, mapping_df


def safe_cuda_cleanup():
    """Безопасная очистка памяти CUDA"""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Ошибка при очистке памяти CUDA: {e}")


def encode_images_streaming(
    image_paths_batch: List[str],
    coordinates: Dict[str, Tuple[float, float]],
    image_encoder,
    transform,
    device,
    max_workers: int = 2
) -> Tuple[List[np.ndarray], List[str], List[Tuple[float, float]]]:
    """Кодирование батча изображений с потоковой обработкой"""
    embeddings = []
    valid_image_paths = []
    valid_coordinates = []
    
    def process_single_image(image_path: str):
        try:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            if image_id not in coordinates:
                return None
                
            image_data = s3_manager.download_bytes(image_path)
            if image_data is None:
                return None
                
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            if image.size[0] < 10 or image.size[1] < 10:
                return None
                
            tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)
            
            with torch.no_grad():
                embedding = image_encoder(tensor).cpu().numpy().flatten()
            
            return embedding, image_path, coordinates[image_id]
            
        except Exception as e:
            logger.debug(f"Ошибка при обработке {image_path}: {e}")
            return None
    
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
    
    if processed_results:
        embeddings, valid_image_paths, valid_coordinates = zip(*processed_results)
        embeddings = list(embeddings)
        valid_image_paths = list(valid_image_paths)
        valid_coordinates = list(valid_coordinates)
    
    safe_cuda_cleanup()
    
    return embeddings, valid_image_paths, valid_coordinates


if __name__ == "__main__":
    main()