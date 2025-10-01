"""
Скрипт для создания CSV файла метаданных из JSON файлов
"""

import sys
from pathlib import Path
import json
import os
import logging
import pandas as pd
from tqdm import tqdm

# Добавляем путь к утилитам
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_metadata_csv(json_dir: str, output_path: str):
    """
    Создает CSV файл с метаданными из JSON файлов
    
    Args:
        json_dir: Директория с JSON файлами
        output_path: Путь для сохранения CSV
    """
    logger.info(f"Создание метаданных из {json_dir}")
    
    if not os.path.exists(json_dir):
        logger.error(f"Директория не найдена: {json_dir}")
        return False
    
    metadata_records = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    logger.info(f"Найдено {len(json_files)} JSON файлов")
    
    for json_file in tqdm(json_files, desc="Обработка JSON файлов"):
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Обрабатываем каждый элемент
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                if 'id' in item and 'geometry' in item:
                    image_id = item['id']
                    
                    # Извлекаем координаты
                    if 'coordinates' in item['geometry']:
                        coords = item['geometry']['coordinates']
                        if len(coords) >= 2:
                            longitude, latitude = coords[0], coords[1]
                            
                            # Создаем полный путь к изображению в S3
                            s3_path = f"site/raw_data/{image_id}.jpeg"
                            
                            record = {
                                'id': image_id,
                                's3_key': s3_path,
                                'latitude': latitude,
                                'longitude': longitude,
                                'captured_at': item.get('captured_at', ''),
                                'compass_angle': item.get('compass_angle', ''),
                                'sequence_id': item.get('sequence', ''),
                                'creator_username': item.get('creator', {}).get('username', ''),
                                'altitude': item.get('altitude', '')
                            }
                            metadata_records.append(record)
                            
        except Exception as e:
            logger.warning(f"Ошибка обработки {json_file}: {e}")
            continue
    
    if not metadata_records:
        logger.error("Не удалось извлечь метаданные")
        return False
    
    # Создаем DataFrame и сохраняем
    df = pd.DataFrame(metadata_records)
    
    # Создаем директорию если её нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Метаданные сохранены в {output_path}")
    logger.info(f"Создано {len(df)} записей")
    logger.info(f"Колонки: {df.columns.tolist()}")
    
    return True

if __name__ == "__main__":
    JSON_DIR = "logs/download_data/cache_moscow_images"
    OUTPUT_CSV = "data/processed/moscow_images.csv"
    
    success = create_metadata_csv(JSON_DIR, OUTPUT_CSV)
    if success:
        print("✅ Метаданные успешно созданы!")
    else:
        print("❌ Ошибка создания метаданных")