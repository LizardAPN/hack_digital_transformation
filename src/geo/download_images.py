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

import concurrent.futures
import csv
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from utils.s3_optimize import S3Manager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class BBoxSplitter:
    """Класс для разбивки bbox на части для лучшего покрытия"""

    @staticmethod
    def split_bbox(bbox: List[float], grid_size: int = 2) -> List[List[float]]:
        """
        Разбивает bbox на grid_size x grid_size частей

        Parameters
        ----------
        bbox : List[float]
            [min_lon, min_lat, max_lon, max_lat]
        grid_size : int, optional
            количество частей по каждой оси (по умолчанию 2)

        Returns
        -------
        List[List[float]]
            Список под-bbox'ов

        Examples
        --------
        >>> bbox = [37.0, 55.0, 38.0, 56.0]
        >>> sub_bboxes = BBoxSplitter.split_bbox(bbox, grid_size=2)
        >>> print(len(sub_bboxes))
        4
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        lon_step = (max_lon - min_lon) / grid_size
        lat_step = (max_lat - min_lat) / grid_size

        sub_bboxes = []

        for i in range(grid_size):
            for j in range(grid_size):
                sub_min_lon = min_lon + (i * lon_step)
                sub_max_lon = min_lon + ((i + 1) * lon_step)
                sub_min_lat = min_lat + (j * lat_step)
                sub_max_lat = min_lat + ((j + 1) * lat_step)

                sub_bboxes.append([sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat])

        return sub_bboxes

    @staticmethod
    def create_bbox_grid(
        center_lat: float, center_lon: float, grid_radius: int = 2, bbox_size: float = 0.02
    ) -> List[List[float]]:
        """
        Создает сетку bbox вокруг центральной точки

        Parameters
        ----------
        center_lat : float
            широта центра
        center_lon : float
            долгота центра
        grid_radius : int, optional
            радиус сетки (количество bbox в каждую сторону от центра) (по умолчанию 2)
        bbox_size : float, optional
            размер каждого bbox в градусах (по умолчанию 0.02)

        Returns
        -------
        List[List[float]]
            Список bbox'ов

        Examples
        --------
        >>> bboxes = BBoxSplitter.create_bbox_grid(55.7558, 37.6176, grid_radius=1)
        >>> print(len(bboxes))
        9
        """
        bboxes = []

        for i in range(-grid_radius, grid_radius + 1):
            for j in range(-grid_radius, grid_radius + 1):
                min_lon = center_lon + (j * bbox_size) - (bbox_size / 2)
                max_lon = center_lon + (j * bbox_size) + (bbox_size / 2)
                min_lat = center_lat + (i * bbox_size) - (bbox_size / 2)
                max_lat = center_lat + (i * bbox_size) + (bbox_size / 2)

                bboxes.append([min_lon, min_lat, max_lon, max_lat])

        return bboxes

    @staticmethod
    def calculate_optimal_grid_size(bbox: List[float], target_bbox_area: float = 0.0004) -> int:
        """
        Рассчитывает оптимальный размер сетки на основе площади bbox

        Parameters
        ----------
        bbox : List[float]
            [min_lon, min_lat, max_lon, max_lat]
        target_bbox_area : float, optional
            целевая площадь каждого под-bbox (в квадратных градусах) (по умолчанию 0.0004)

        Returns
        -------
        int
            Оптимальный размер сетки

        Examples
        --------
        >>> bbox = [37.0, 55.0, 38.0, 56.0]
        >>> grid_size = BBoxSplitter.calculate_optimal_grid_size(bbox)
        >>> print(grid_size)
        50
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_area = (max_lon - min_lon) * (max_lat - min_lat)

        # Рассчитываем количество частей так, чтобы площадь каждого была около target_bbox_area
        grid_size = max(1, int((bbox_area / target_bbox_area) ** 0.5))

        # Ограничиваем максимальный размер сетки для избежания слишком маленьких bbox
        return min(grid_size, 10)


class MapillaryS3Client:
    """Mapillary клиент с прямой загрузкой в S3"""

    def __init__(
        self,
        access_token: str,
        s3_manager: S3Manager,
        max_workers: int = 10,
        cache_dir: Optional[str] = None,
    ):
        """
        Инициализация MapillaryS3Client

        Parameters
        ----------
        access_token : str
            Токен доступа к Mapillary API
        s3_manager : S3Manager
            Менеджер для работы с S3
        max_workers : int, optional
            Максимальное количество рабочих потоков (по умолчанию 10)
        cache_dir : str, optional
            Директория для кэширования данных (по умолчанию None)

        Examples
        --------
        >>> s3_manager = S3Manager()
        >>> client = MapillaryS3Client(
        ...     access_token="your_token",
        ...     s3_manager=s3_manager,
        ...     max_workers=5,
        ...     cache_dir="/tmp/mapillary_cache"
        ... )
        """
        self.access_token = access_token
        self.s3_manager = s3_manager
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.base_url = "https://graph.mapillary.com"
        self.request_lock = Lock()
        self.bbox_splitter = BBoxSplitter()

        # Настройка HTTP сессии
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept": "application/json"}
        )

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_cached_data(self, key: str, expiry_hours: int = 24) -> Optional[dict]:
        """Кэширование данных API запросов"""
        if not self.cache_dir:
            return None

        hash_key = hashlib.md5(key.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.json")

        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < (expiry_hours * 3600):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Ошибка чтения кэша: {e}")
                    return None
        return None

    def _set_cached_data(self, key: str, data: dict):
        """Сохранение данных в кэш"""
        if not self.cache_dir:
            return

        hash_key = hashlib.md5(key.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.json")

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")

    def _rate_limited_request(self, url: str, params: dict = None) -> dict:
        """HTTP запрос с ограничением скорости и повторными попытками"""
        with self.request_lock:
            time.sleep(0.1)  # Базовая задержка между запросами

        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == 2:  # Последняя попытка
                    raise e
                time.sleep(2**attempt)  # Экспоненциальная задержка

    def get_images_in_bbox(self, bbox: List[float], max_results: int = 1000, use_cache: bool = True) -> List[Dict]:
        """
        Получение списка изображений в bounding box
        bbox: [min_lon, min_lat, max_lon, max_lat]
        """
        cache_key = f"bbox_{'_'.join(map(str, bbox))}_{max_results}"

        if use_cache:
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                logger.info("Используются кэшированные данные")
                return cached_data

        url = f"{self.base_url}/images"
        params = {
            "fields": "id,geometry,compass_angle,captured_at,altitude,sequence,creator,thumb_2048_url",
            "access_token": self.access_token,
            "limit": min(max_results, 2000),
            "bbox": ",".join(map(str, bbox)),
        }

        try:
            data = self._rate_limited_request(url, params)

            if "data" in data:
                self._set_cached_data(cache_key, data["data"])
                return data["data"]
            else:
                logger.error(f"Неожиданный формат ответа: {data.get('error', 'Unknown error')}")
                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса для bbox {bbox}: {e}")
            return []

    def get_images_in_bbox_batch(
        self, bboxes: List[List[float]], max_results_per_bbox: int = 500, use_cache: bool = True
    ) -> List[Dict]:
        """
        Пакетное получение изображений для нескольких bbox

        Parameters
        ----------
        bboxes : List[List[float]]
            список bbox'ов
        max_results_per_bbox : int, optional
            максимальное количество результатов на bbox (по умолчанию 500)
        use_cache : bool, optional
            использовать кэширование (по умолчанию True)

        Returns
        -------
        List[Dict]
            Объединенный список уникальных изображений

        Examples
        --------
        >>> client = MapillaryS3Client(access_token="token", s3_manager=s3_manager)
        >>> bboxes = [[37.0, 55.0, 37.5, 55.5], [37.5, 55.0, 38.0, 55.5]]
        >>> images = client.get_images_in_bbox_batch(bboxes)
        >>> print(len(images))
        """
        all_images = []

        logger.info(f"Начинаем обработку {len(bboxes)} bbox регионов...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(bboxes))) as executor:
            future_to_bbox = {
                executor.submit(self.get_images_in_bbox, bbox, max_results_per_bbox, use_cache): bbox for bbox in bboxes
            }

            for i, future in enumerate(concurrent.futures.as_completed(future_to_bbox)):
                bbox = future_to_bbox[future]
                try:
                    images = future.result()
                    all_images.extend(images)
                    logger.info(f"Обработан bbox {i+1}/{len(bboxes)}: {bbox} - найдено {len(images)} изображений")
                except Exception as e:
                    logger.error(f"Ошибка для bbox {bbox}: {e}")

        # Удаляем дубликаты по ID
        seen_ids = set()
        unique_images = []

        for img in all_images:
            img_id = img.get("id")
            if img_id and img_id not in seen_ids:
                seen_ids.add(img_id)
                unique_images.append(img)

        logger.info(f"После объединения: {len(unique_images)} уникальных изображений")
        return unique_images

    def get_images_for_large_area(
        self,
        bbox: List[float],
        grid_size: Optional[int] = None,
        max_results_per_bbox: int = 500,
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        Получение изображений для большого региона с автоматическим разбиением на части

        Parameters
        ----------
        bbox : List[float]
            основной bbox [min_lon, min_lat, max_lon, max_lat]
        grid_size : int, optional
            размер сетки (если None, рассчитывается автоматически) (по умолчанию None)
        max_results_per_bbox : int, optional
            максимальное количество результатов на под-bbox (по умолчанию 500)
        use_cache : bool, optional
            использовать кэширование (по умолчанию True)

        Returns
        -------
        List[Dict]
            Список уникальных изображений

        Examples
        --------
        >>> client = MapillaryS3Client(access_token="token", s3_manager=s3_manager)
        >>> bbox = [37.0, 55.0, 38.0, 56.0]
        >>> images = client.get_images_for_large_area(bbox)
        >>> print(len(images))
        """
        # Рассчитываем оптимальный размер сетки если не задан
        if grid_size is None:
            grid_size = self.bbox_splitter.calculate_optimal_grid_size(bbox)
            logger.info(f"Автоматически рассчитан размер сетки: {grid_size}x{grid_size}")

        # Разбиваем основной bbox на части
        sub_bboxes = self.bbox_splitter.split_bbox(bbox, grid_size)

        logger.info(f"Основной bbox разбит на {len(sub_bboxes)} частей")
        for i, sub_bbox in enumerate(sub_bboxes):
            logger.info(f"  Часть {i+1}: {sub_bbox}")

        # Получаем изображения для всех частей
        return self.get_images_in_bbox_batch(sub_bboxes, max_results_per_bbox, use_cache)

    def get_images_around_point(
        self,
        center_lat: float,
        center_lon: float,
        grid_radius: int = 2,
        bbox_size: float = 0.02,
        max_results_per_bbox: int = 500,
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        Получение изображений вокруг центральной точки

        Parameters
        ----------
        center_lat : float
            широта центра
        center_lon : float
            долгота центра
        grid_radius : int, optional
            радиус сетки (по умолчанию 2)
        bbox_size : float, optional
            размер каждого bbox в градусах (по умолчанию 0.02)
        max_results_per_bbox : int, optional
            максимальное количество результатов на bbox (по умолчанию 500)
        use_cache : bool, optional
            использовать кэширование (по умолчанию True)

        Returns
        -------
        List[Dict]
            Список уникальных изображений

        Examples
        --------
        >>> client = MapillaryS3Client(access_token="token", s3_manager=s3_manager)
        >>> images = client.get_images_around_point(55.7558, 37.6176)
        >>> print(len(images))
        """
        # Создаем сетку bbox вокруг точки
        bboxes = self.bbox_splitter.create_bbox_grid(center_lat, center_lon, grid_radius, bbox_size)

        logger.info(f"Создана сетка из {len(bboxes)} bbox вокруг точки ({center_lat}, {center_lon})")

        # Получаем изображения для всех bbox
        return self.get_images_in_bbox_batch(bboxes, max_results_per_bbox, use_cache)

    def _prepare_s3_metadata(self, image_info: Dict) -> Dict[str, str]:
        """Подготовка метаданных для S3 (все значения должны быть строками)"""
        geometry = image_info.get("geometry", {})
        coordinates = geometry.get("coordinates", [0, 0])

        # ВАЖНО: Все значения должны быть строками для S3 metadata
        metadata = {
            "mapillary_id": str(image_info.get("id", "")),
            "latitude": str(coordinates[1]) if coordinates else "0",
            "longitude": str(coordinates[0]) if coordinates else "0",
            "captured_at": str(image_info.get("captured_at", "")),
            "compass_angle": str(image_info.get("compass_angle", "")),
            "altitude": str(image_info.get("altitude", "")),
            "upload_timestamp": datetime.now().isoformat(),
        }

        # Добавляем опциональные поля
        sequence = image_info.get("sequence")
        if sequence and isinstance(sequence, dict):
            metadata["sequence_id"] = str(sequence.get("id", ""))

        creator = image_info.get("creator")
        if creator and isinstance(creator, dict):
            metadata["creator_username"] = str(creator.get("username", ""))

        return metadata

    def download_and_upload_to_s3(
        self,
        images: List[Dict],
        s3_prefix: str,
        image_format: str = "JPEG",
        quality: int = 85,
        progress_callback: callable = None,
    ) -> Dict:
        """Прямая загрузка изображений с Mapillary в S3"""
        results = {"successful": 0, "failed": 0, "skipped": 0, "errors": []}

        def _process_image(image_info: Dict) -> tuple:
            """Обработка одного изображения"""
            image_id = image_info.get("id")
            s3_key = f"{s3_prefix}/{image_id}.{image_format.lower()}"

            try:
                # Проверяем, существует ли файл в S3
                if self.s3_manager.file_exists(s3_key):
                    return image_id, False, "Файл уже существует в S3"

                # Получаем URL изображения
                image_url = image_info.get("thumb_2048_url")
                if not image_url:
                    image_url = self._get_image_url(image_id)

                if not image_url:
                    return image_id, False, "URL изображения не найден"

                # Скачиваем изображение
                response = self.session.get(image_url, timeout=30)
                response.raise_for_status()

                # Подготавливаем метаданные для S3
                metadata = self._prepare_s3_metadata(image_info)

                # Загружаем напрямую в S3
                success = self.s3_manager.upload_image_data(
                    response.content, s3_key, image_format=image_format, quality=quality, metadata=metadata
                )

                if success:
                    return image_id, True, "Успешно"
                else:
                    return image_id, False, "Ошибка загрузки в S3"

            except Exception as e:
                return image_id, False, f"Ошибка: {str(e)}"

        # Многопоточная обработка изображений
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {executor.submit(_process_image, img): img for img in images}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_image)):
                image_info = future_to_image[future]
                image_id, success, message = future.result()

                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{image_id}: {message}")

                # Отчет о прогрессе
                self._report_progress(i, len(images), results, progress_callback)

        # Добавляем статистику S3
        results["s3_stats"] = self.s3_manager.get_upload_stats()

        logger.info(f"Загрузка завершена. Успешно: {results['successful']}, " f"Ошибки: {results['failed']}")

        return results

    def _get_image_url(self, image_id: str) -> str:
        """Получение URL изображения по ID"""
        try:
            url = f"{self.base_url}/{image_id}"
            params = {"fields": "thumb_2048_url", "access_token": self.access_token}

            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            return data.get("thumb_2048_url", "")
        except Exception as e:
            logger.error(f"Ошибка получения URL для {image_id}: {e}")
            return ""

    def _report_progress(self, current: int, total: int, results: Dict, progress_callback: callable = None):
        """Отчет о прогрессе обработки"""
        if (current + 1) % 10 == 0 or (current + 1) == total:
            processed = results["successful"] + results["failed"] + results["skipped"]
            logger.info(
                f"Прогресс: {processed}/{total} | "
                f"Успешно: {results['successful']} | "
                f"Ошибки: {results['failed']}"
            )

        if progress_callback:
            progress_callback(current + 1, total)

    def _safe_get(self, obj, key, default=""):
        """Безопасное получение значения из объекта"""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    def save_metadata_to_csv(self, images: List[Dict], filename: str, include_extended: bool = True):
        """Сохранение метаданных в CSV файл"""
        if not images:
            logger.warning("Нет данных для сохранения")
            return

        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

        # Определяем поля для CSV
        fieldnames = ["id", "latitude", "longitude", "captured_at", "compass_angle"]
        if include_extended:
            fieldnames.extend(["sequence_id", "creator_username", "altitude"])

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for img in images:
                # Проверяем, что это словарь
                if not isinstance(img, dict):
                    logger.warning(f"Пропущен некорректный элемент: {type(img)}")
                    continue

                # Безопасно извлекаем geometry
                geometry = self._safe_get(img, "geometry", {})
                coordinates = self._safe_get(geometry, "coordinates", [0, 0])

                # Безопасно извлекаем основные поля
                row = {
                    "id": self._safe_get(img, "id"),
                    "latitude": coordinates[1] if isinstance(coordinates, list) and len(coordinates) > 1 else 0,
                    "longitude": coordinates[0] if isinstance(coordinates, list) and len(coordinates) > 0 else 0,
                    "captured_at": self._safe_get(img, "captured_at"),
                    "compass_angle": self._safe_get(img, "compass_angle"),
                }

                if include_extended:
                    # Безопасно извлекаем расширенные поля
                    sequence = self._safe_get(img, "sequence")
                    creator = self._safe_get(img, "creator")

                    row.update(
                        {
                            "sequence_id": self._safe_get(sequence, "id") if isinstance(sequence, dict) else "",
                            "creator_username": (
                                self._safe_get(creator, "username") if isinstance(creator, dict) else ""
                            ),
                            "altitude": self._safe_get(img, "altitude"),
                        }
                    )

                writer.writerow(row)

        logger.info(f"Метаданные сохранены в {filename} ({len(images)} записей)")
