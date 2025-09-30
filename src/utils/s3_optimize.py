import concurrent.futures
import io
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class S3Manager:
    """
    Менеджер для высокопроизводительной работы с S3 (загрузка и скачивание).

    Класс предоставляет интерфейс для эффективной работы с облачным хранилищем S3,
    включая параллельную загрузку и скачивание файлов, управление соединениями
    и сбор статистики операций.

    Атрибуты
    ----------
    key_id : str
        Идентификатор ключа доступа к S3.
    access_key : str
        Секретный ключ доступа к S3.
    endpoint_url : str
        URL конечной точки S3.
    bucket_name : str
        Имя бакета S3.
    max_workers : int
        Максимальное количество рабочих потоков для параллельных операций.
    chunk_size : int
        Размер части файла для multipart операций в байтах.
    """

    def __init__(
        self,
        key_id: str | None = None,
        access_key: str | None = None,
        endpoint_url: str | None = None,
        bucket_name: str | None = None,
        max_workers: int = 10,
        chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks for multipart
    ):
        """
        Инициализация менеджера S3.

        Параметры
        ----------
        key_id : str, optional
            Идентификатор ключа доступа к S3. Если не указан, берется из переменной окружения AWS_ACCESS_KEY_ID.
        access_key : str, optional
            Секретный ключ доступа к S3. Если не указан, берется из переменной окружения AWS_SECRET_ACCESS_KEY.
        endpoint_url : str, optional
            URL конечной точки S3. Если не указан, берется из переменной окружения AWS_ENDPOINT_URL 
            или используется значение по умолчанию "https://s3-msk.tinkoff.ru".
        bucket_name : str, optional
            Имя бакета S3. Если не указан, берется из переменной окружения AWS_BUCKET_NAME.
        max_workers : int, optional
            Максимальное количество рабочих потоков для параллельных операций (по умолчанию 10).
        chunk_size : int, optional
            Размер части файла для multipart операций в байтах (по умолчанию 8MB).
        
        Исключения
        ----------
        ValueError
            Возникает, если не указаны обязательные параметры подключения к S3.
        """
        self.key_id = key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.access_key = access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL", "https://s3-msk.tinkoff.ru")
        self.bucket_name = bucket_name or os.getenv("AWS_BUCKET_NAME")
        self.max_workers = max_workers
        self.chunk_size = chunk_size

        if not all([self.key_id, self.access_key, self.bucket_name]):
            raise ValueError("Missing required AWS credentials")

        self._client = None
        self._upload_lock = threading.Lock()
        self._download_lock = threading.Lock()
        self._upload_stats = {"successful": 0, "failed": 0, "total_size": 0}
        self._download_stats = {"successful": 0, "failed": 0, "total_size": 0}

    @property
    def client(self):
        """
        Ленивая инициализация S3 клиента.

        Создает и возвращает клиент S3 при первом обращении. При последующих обращениях
        возвращает уже созданный клиент. Также проверяет подключение к бакету.

        Возвращает
        -------
        boto3.Client
            Инициализированный клиент S3.

        Исключения
        ----------
        Exception
            Возникает, если не удалось создать клиент или подключиться к бакету.
        """
        if self._client is None:
            try:
                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=self.key_id,
                    aws_secret_access_key=self.access_key,
                    endpoint_url=self.endpoint_url,
                )
                # Проверяем подключение
                self.client.head_bucket(Bucket=self.bucket_name)
                logger.info(f"Успешно подключились к S3 бакету: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Ошибка при создании S3 клиента: {e}")
                raise
        return self._client

    # ========== МЕТОДЫ ДЛЯ ЗАГРУЗКИ В S3 ==========

    def upload_bytes_parallel(
        self, data: bytes, s3_key: str, content_type: str = "application/octet-stream", metadata: Optional[Dict] = None
    ) -> bool:
        """
        Параллельная загрузка больших файлов с использованием multipart upload.

        Для файлов размером больше chunk_size выполняет параллельную загрузку частями.
        Для маленьких файлов использует обычную загрузку.

        Параметры
        ----------
        data : bytes
            Данные для загрузки в виде байтов.
        s3_key : str
            Ключ объекта в S3, по которому будет сохранен файл.
        content_type : str, optional
            MIME-тип контента (по умолчанию "application/octet-stream").
        metadata : dict, optional
            Дополнительные метаданные для объекта (по умолчанию None).

        Возвращает
        -------
        bool
            True, если загрузка прошла успешно, иначе False.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> data = b"some binary data"
        >>> success = s3_manager.upload_bytes_parallel(data, "path/to/file.bin")
        >>> print(success)
        True
        """
        try:
            if len(data) <= self.chunk_size:
                # Для маленьких файлов используем обычную загрузку
                return self.upload_bytes(data, s3_key, content_type, metadata)

            # Создаем multipart upload
            multipart_args = {"ContentType": content_type}
            if metadata:
                multipart_args["Metadata"] = metadata

            mpu = self.client.create_multipart_upload(Bucket=self.bucket_name, Key=s3_key, **multipart_args)
            mpu_id = mpu["UploadId"]

            parts = []
            futures = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Разбиваем данные на части и загружаем параллельно
                for i, start in enumerate(range(0, len(data), self.chunk_size)):
                    end = min(start + self.chunk_size, len(data))
                    part_data = data[start:end]

                    future = executor.submit(self._upload_part, mpu_id, s3_key, i + 1, part_data)
                    futures.append(future)

                # Собираем результаты
                for future in concurrent.futures.as_completed(futures):
                    part_num, etag = future.result()
                    parts.append({"PartNumber": part_num, "ETag": etag})

            # Завершаем multipart upload
            parts.sort(key=lambda x: x["PartNumber"])
            self.client.complete_multipart_upload(
                Bucket=self.bucket_name, Key=s3_key, UploadId=mpu_id, MultipartUpload={"Parts": parts}
            )

            with self._upload_lock:
                self._upload_stats["successful"] += 1
                self._upload_stats["total_size"] += len(data)

            logger.info(f"Multipart upload завершен: {s3_key} ({len(data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Ошибка multipart upload для {s3_key}: {e}")
            with self._upload_lock:
                self._upload_stats["failed"] += 1
            return False

    def _upload_part(self, upload_id: str, s3_key: str, part_number: int, data: bytes) -> tuple:
        """
        Загрузка одной части в multipart upload.

        Внутренний метод, используемый для загрузки отдельной части файла
        в рамках multipart upload операции.

        Параметры
        ----------
        upload_id : str
            Идентификатор multipart upload сессии.
        s3_key : str
            Ключ объекта в S3.
        part_number : int
            Номер части файла (начиная с 1).
        data : bytes
            Данные части файла для загрузки.

        Возвращает
        -------
        tuple
            Кортеж из номера части и ETag ответа от S3.

        Исключения
        ----------
        Exception
            Возникает, если произошла ошибка при загрузке части файла.
        """
        try:
            response = self.client.upload_part(
                Bucket=self.bucket_name, Key=s3_key, PartNumber=part_number, UploadId=upload_id, Body=io.BytesIO(data)
            )
            return part_number, response["ETag"]
        except Exception as e:
            logger.error(f"Ошибка загрузки части {part_number} для {s3_key}: {e}")
            raise

    def upload_bytes(
        self, data: bytes, s3_key: str, content_type: str = "application/octet-stream", metadata: Optional[Dict] = None
    ) -> bool:
        """
        Быстрая загрузка байтов в S3.

        Загружает данные в виде байтов в указанный объект S3. Подходит для
        относительно небольших файлов, которые не требуют multipart загрузки.

        Параметры
        ----------
        data : bytes
            Данные для загрузки в виде байтов.
        s3_key : str
            Ключ объекта в S3, по которому будет сохранен файл.
        content_type : str, optional
            MIME-тип контента (по умолчанию "application/octet-stream").
        metadata : dict, optional
            Дополнительные метаданные для объекта (по умолчанию None).

        Возвращает
        -------
        bool
            True, если загрузка прошла успешно, иначе False.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> data = b"some binary data"
        >>> success = s3_manager.upload_bytes(data, "path/to/file.bin")
        >>> print(success)
        True
        """
        try:
            extra_args = {"ContentType": content_type}
            if metadata:
                extra_args["Metadata"] = metadata

            self.client.upload_fileobj(io.BytesIO(data), self.bucket_name, s3_key, ExtraArgs=extra_args)

            with self._upload_lock:
                self._upload_stats["successful"] += 1
                self._upload_stats["total_size"] += len(data)

            logger.debug(f"Успешная загрузка: {s3_key} ({len(data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки байтов в {s3_key}: {e}")
            with self._upload_lock:
                self._upload_stats["failed"] += 1
            return False

    def upload_file_parallel(
        self, file_path: str, s3_key: str, remove_source: bool = False, metadata: Optional[Dict] = None
    ) -> bool:
        """
        Параллельная загрузка файла с оптимизацией для больших файлов.

        Загружает файл из локальной файловой системы в S3. Для больших файлов
        использует multipart загрузку, для маленьких - стандартную загрузку.
        При необходимости может удалить исходный файл после успешной загрузки.

        Параметры
        ----------
        file_path : str
            Путь к локальному файлу для загрузки.
        s3_key : str
            Ключ объекта в S3, по которому будет сохранен файл.
        remove_source : bool, optional
            Флаг, указывающий на необходимость удаления исходного файла
            после успешной загрузки (по умолчанию False).
        metadata : dict, optional
            Дополнительные метаданные для объекта (по умолчанию None).

        Возвращает
        -------
        bool
            True, если загрузка прошла успешно, иначе False.

        Примеры
        --------
        >>> s3_manager = S3Manager()
s        >>> success = s3_manager.upload_file_parallel("/path/to/local/file.txt", "path/in/s3/file.txt")
        >>> print(success)
        True
        """
        try:
            file_size = os.path.getsize(file_path)
            content_type = self._get_content_type(file_path)

            if file_size > self.chunk_size:
                # Для больших файлов читаем и загружаем через multipart
                with open(file_path, "rb") as f:
                    data = f.read()
                result = self.upload_bytes_parallel(data, s3_key, content_type, metadata)
            else:
                # Для маленьких файлов используем стандартную загрузку
                extra_args = {"ContentType": content_type}
                if metadata:
                    extra_args["Metadata"] = metadata

                self.client.upload_file(file_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
                result = True
                with self._upload_lock:
                    self._upload_stats["successful"] += 1
                    self._upload_stats["total_size"] += file_size

            if result and remove_source:
                os.remove(file_path)
                logger.debug(f"Исходный файл удален: {file_path}")

            return result

        except Exception as e:
            logger.error(f"Ошибка загрузки файла {file_path} в {s3_key}: {e}")
            with self._upload_lock:
                self._upload_stats["failed"] += 1
            return False

    def batch_upload_files(
        self, file_paths: List[str], s3_prefix: str = "", remove_source: bool = False, progress_callback=None
    ) -> Dict:
        """
        Пакетная загрузка файлов с многопоточностью.

        Загружает несколько файлов параллельно, используя многопоточность для повышения
        производительности. Каждый файл загружается независимо с использованием
        метода upload_file_parallel.

        Параметры
        ----------
        file_paths : List[str]
            Список путей к локальным файлам для загрузки.
        s3_prefix : str, optional
            Префикс для ключей объектов в S3 (по умолчанию "").
        remove_source : bool, optional
            Флаг, указывающий на необходимость удаления исходных файлов
            после успешной загрузки (по умолчанию False).
        progress_callback : callable, optional
            Функция обратного вызова для отслеживания прогресса загрузки (по умолчанию None).

        Возвращает
        -------
        Dict
            Словарь с результатами загрузки, содержащий ключи:
            - "successful": список успешно загруженных файлов
            - "failed": список кортежей (файл, сообщение об ошибке) для неудачных загрузок
            - "total": общее количество файлов для загрузки

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> file_paths = ["/path/to/file1.txt", "/path/to/file2.txt"]
        >>> results = s3_manager.batch_upload_files(file_paths, "uploads/")
        >>> print(f"Успешно загружено: {len(results['successful'])}")
        >>> print(f"Ошибок: {len(results['failed'])}")
        """
        results = {"successful": [], "failed": [], "total": len(file_paths)}

        def _upload_file(file_path):
            try:
                filename = Path(file_path).name
                s3_key = f"{s3_prefix}/{filename}" if s3_prefix else filename

                if self.upload_file_parallel(file_path, s3_key, remove_source):
                    return file_path, True, s3_key
                else:
                    return file_path, False, s3_key

            except Exception as e:
                return file_path, False, str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(_upload_file, file_path): file_path for file_path in file_paths}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_path, success, message = future.result()

                if success:
                    results["successful"].append(file_path)
                else:
                    results["failed"].append((file_path, message))

                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, len(file_paths))

        logger.info(f"Пакетная загрузка завершена: {len(results['successful'])}/{len(file_paths)} успешно")
        return results

    def upload_image_data(
        self,
        image_data: Union[bytes, BinaryIO],
        s3_key: str,
        image_format: str = "JPEG",
        quality: int = 85,
        optimize: bool = True,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Оптимизированная загрузка изображений с возможностью сжатия.

        Загружает изображение в S3 с возможностью оптимизации и сжатия. Поддерживает
        различные форматы изображений (JPEG, PNG, GIF, WEBP) и автоматически
        конвертирует изображения в RGB при необходимости.

        Параметры
        ----------
        image_data : Union[bytes, BinaryIO]
            Данные изображения в виде байтов или файлового объекта.
        s3_key : str
            Ключ объекта в S3, по которому будет сохранено изображение.
        image_format : str, optional
            Формат изображения для сохранения (по умолчанию "JPEG").
        quality : int, optional
            Качество изображения для форматов с потерями (по умолчанию 85).
        optimize : bool, optional
            Флаг оптимизации изображения (по умолчанию True).
        metadata : dict, optional
            Дополнительные метаданные для объекта (по умолчанию None).

        Возвращает
        -------
        bool
            True, если загрузка прошла успешно, иначе False.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> with open("image.jpg", "rb") as f:
        ...     image_data = f.read()
        >>> success = s3_manager.upload_image_data(image_data, "images/optimized.jpg", quality=90)
        >>> print(success)
        True
        """
        try:
            import io

            from PIL import Image

            content_types = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}

            content_type = content_types.get(image_format.upper(), "image/jpeg")

            # Если данные уже в bytes, оптимизируем их
            if isinstance(image_data, bytes):
                with Image.open(io.BytesIO(image_data)) as img:
                    output = io.BytesIO()

                    # Конвертируем в RGB если нужно
                    if image_format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")

                    save_args = {"format": image_format, "optimize": optimize}
                    if image_format.upper() == "JPEG":
                        save_args["quality"] = quality
                    elif image_format.upper() == "WEBP":
                        save_args["quality"] = quality

                    img.save(output, **save_args)
                    processed_data = output.getvalue()
            else:
                # Если это файловый объект
                image_data.seek(0)
                with Image.open(image_data) as img:
                    output = io.BytesIO()

                    if image_format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")

                    save_args = {"format": image_format, "optimize": optimize}
                    if image_format.upper() == "JPEG":
                        save_args["quality"] = quality

                    img.save(output, **save_args)
                    processed_data = output.getvalue()

            # Добавляем метаданные об оптимизации
            if metadata is None:
                metadata = {}
            # Убедимся, что все значения метаданных - строки
            safe_metadata = {}
            for k, v in metadata.items():
                safe_metadata[k] = str(v)

            safe_metadata["image_optimized"] = "true"
            safe_metadata["image_format"] = image_format
            safe_metadata["image_quality"] = str(quality)

            return self.upload_bytes(processed_data, s3_key, content_type, safe_metadata)

        except ImportError:
            logger.warning("PIL не установлен, используем базовую загрузку")
            if hasattr(image_data, "read"):
                image_data.seek(0)
                data = image_data.read()
            else:
                data = image_data

            return self.upload_bytes(data, s3_key, "image/jpeg", metadata)
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения {s3_key}: {e}")
            return False

    # ========== МЕТОДЫ ДЛЯ СКАЧИВАНИЯ ИЗ S3 ==========

    def download_bytes(self, s3_key: str) -> Optional[bytes]:
        """
        Загрузка файла из S3 в виде байтов.

        Загружает объект из S3 и возвращает его содержимое в виде байтов.
        Подходит для относительно небольших файлов, которые помещаются в память.

        Параметры
        ----------
        s3_key : str
            Ключ объекта в S3 для загрузки.

        Возвращает
        -------
        Optional[bytes]
            Содержимое файла в виде байтов, или None в случае ошибки.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> data = s3_manager.download_bytes("path/to/file.bin")
        >>> if data is not None:
        ...     print(f"Размер загруженных данных: {len(data)} байт")
        """
        try:
            with io.BytesIO() as data_stream:
                self.client.download_fileobj(self.bucket_name, s3_key, data_stream)
                data = data_stream.getvalue()

            with self._download_lock:
                self._download_stats["successful"] += 1
                self._download_stats["total_size"] += len(data)

            logger.debug(f"Успешная загрузка: {s3_key} ({len(data)} bytes)")
            return data

        except Exception as e:
            logger.error(f"Ошибка загрузки байтов из {s3_key}: {e}")
            with self._download_lock:
                self._download_stats["failed"] += 1
            return None

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Загрузка файла из S3 в локальный файл.

        Загружает объект из S3 и сохраняет его в локальный файл. Автоматически
        создает необходимые директории в пути к файлу.

        Параметры
        ----------
        s3_key : str
            Ключ объекта в S3 для загрузки.
        local_path : str
            Путь к локальному файлу для сохранения данных.

        Возвращает
        -------
        bool
            True, если загрузка прошла успешно, иначе False.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> success = s3_manager.download_file("path/to/remote/file.txt", "/local/path/file.txt")
        >>> if success:
        ...     print("Файл успешно загружен")
        """
        try:
            # Создаем директории если нужно
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.client.download_file(self.bucket_name, s3_key, local_path)

            file_size = os.path.getsize(local_path)
            with self._download_lock:
                self._download_stats["successful"] += 1
                self._download_stats["total_size"] += file_size

            logger.debug(f"Успешная загрузка: {s3_key} -> {local_path} ({file_size} bytes)")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки файла {s3_key} в {local_path}: {e}")
            with self._download_lock:
                self._download_stats["failed"] += 1
            return False

    def download_bytes_parallel(self, s3_key: str) -> Optional[bytes]:
        """
        Параллельная загрузка больших файлов с использованием multipart download.

        Для файлов размером больше chunk_size выполняет параллельную загрузку частями.
        Для маленьких файлов использует обычную загрузку.

        Параметры
        ----------
        s3_key : str
            Ключ объекта в S3 для загрузки.

        Возвращает
        -------
        Optional[bytes]
            Содержимое файла в виде байтов, или None в случае ошибки.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> data = s3_manager.download_bytes_parallel("path/to/large_file.bin")
        >>> if data is not None:
        ...     print(f"Размер загруженных данных: {len(data)} байт")
        """
        try:
            # Получаем размер файла
            head_response = self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            file_size = head_response["ContentLength"]

            if file_size <= self.chunk_size:
                # Для маленьких файлов используем обычную загрузку
                return self.download_bytes(s3_key)

            futures = []
            parts = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Загружаем части файла параллельно
                for i, start_byte in enumerate(range(0, file_size, self.chunk_size)):
                    end_byte = min(start_byte + self.chunk_size - 1, file_size - 1)

                    future = executor.submit(self._download_part, s3_key, start_byte, end_byte, i)
                    futures.append(future)

                # Собираем результаты
                for future in concurrent.futures.as_completed(futures):
                    part_data, part_num = future.result()
                    parts.append((part_num, part_data))

            # Собираем файл из частей
            parts.sort(key=lambda x: x[0])
            full_data = b"".join(part_data for _, part_data in parts)

            with self._download_lock:
                self._download_stats["successful"] += 1
                self._download_stats["total_size"] += len(full_data)

            logger.info(f"Multipart download завершен: {s3_key} ({len(full_data)} bytes)")
            return full_data

        except Exception as e:
            logger.error(f"Ошибка multipart download для {s3_key}: {e}")
            with self._download_lock:
                self._download_stats["failed"] += 1
            return None

    def _download_part(self, s3_key: str, start_byte: int, end_byte: int, part_num: int) -> tuple:
        """
        Загрузка одной части файла.

        Внутренний метод, используемый для загрузки отдельной части файла
        в рамках multipart download операции.

        Параметры
        ----------
        s3_key : str
            Ключ объекта в S3.
        start_byte : int
            Начальная позиция байта для загрузки.
        end_byte : int
            Конечная позиция байта для загрузки.
        part_num : int
            Номер части файла.

        Возвращает
        -------
        tuple
            Кортеж из данных части файла и номера части.

        Исключения
        ----------
        Exception
            Возникает, если произошла ошибка при загрузке части файла.
        """
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name, Key=s3_key, Range=f"bytes={start_byte}-{end_byte}"
            )
            part_data = response["Body"].read()
            return part_data, part_num
        except Exception as e:
            logger.error(f"Ошибка загрузки части {part_num} для {s3_key}: {e}")
            raise

    def batch_download_files(self, s3_keys: List[str], local_dir: str = "", progress_callback=None) -> Dict:
        """
        Пакетная загрузка файлов с многопоточностью.

        Загружает несколько файлов параллельно из S3 в локальную файловую систему,
        используя многопоточность для повышения производительности. Каждый файл
        загружается независимо с использованием метода download_file.

        Параметры
        ----------
        s3_keys : List[str]
            Список ключей объектов в S3 для загрузки.
        local_dir : str, optional
            Локальная директория для сохранения файлов (по умолчанию "").
        progress_callback : callable, optional
            Функция обратного вызова для отслеживания прогресса загрузки (по умолчанию None).

        Возвращает
        -------
        Dict
            Словарь с результатами загрузки, содержащий ключи:
            - "successful": список успешно загруженных файлов
            - "failed": список кортежей (ключ, сообщение об ошибке) для неудачных загрузок
            - "total": общее количество файлов для загрузки

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> s3_keys = ["path/to/file1.txt", "path/to/file2.txt"]
        >>> results = s3_manager.batch_download_files(s3_keys, "/local/downloads/")
        >>> print(f"Успешно загружено: {len(results['successful'])}")
        >>> print(f"Ошибок: {len(results['failed'])}")
        """
        results = {"successful": [], "failed": [], "total": len(s3_keys)}

        def _download_file(s3_key):
            try:
                # Определяем локальный путь
                if local_dir:
                    filename = Path(s3_key).name
                    local_path = os.path.join(local_dir, filename)
                else:
                    local_path = Path(s3_key).name

                if self.download_file(s3_key, local_path):
                    return s3_key, True, local_path
                else:
                    return s3_key, False, local_path

            except Exception as e:
                return s3_key, False, str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {executor.submit(_download_file, s3_key): s3_key for s3_key in s3_keys}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_key)):
                s3_key, success, message = future.result()

                if success:
                    results["successful"].append(s3_key)
                else:
                    results["failed"].append((s3_key, message))

                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, len(s3_keys))

        logger.info(f"Пакетная загрузка завершена: {len(results['successful'])}/{len(s3_keys)} успешно")
        return results

    def batch_download_bytes(self, s3_keys: List[str], progress_callback=None) -> Dict[str, Optional[bytes]]:
        """
        Пакетная загрузка файлов в память с многопоточностью.

        Загружает несколько файлов параллельно из S3 в память, используя
        многопоточность для повышения производительности. Каждый файл
        загружается независимо с использованием метода download_bytes.

        Параметры
        ----------
        s3_keys : List[str]
            Список ключей объектов в S3 для загрузки.
        progress_callback : callable, optional
            Функция обратного вызова для отслеживания прогресса загрузки (по умолчанию None).

        Возвращает
        -------
        Dict[str, Optional[bytes]]
            Словарь, где ключи - это S3 ключи, а значения - данные файлов в виде байтов
            или None в случае ошибки загрузки конкретного файла.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> s3_keys = ["path/to/file1.txt", "path/to/file2.txt"]
        >>> results = s3_manager.batch_download_bytes(s3_keys)
        >>> for key, data in results.items():
        ...     if data is not None:
        ...         print(f"{key}: {len(data)} bytes")
        ...     else:
        ...         print(f"Ошибка загрузки {key}")
        """
        results = {}

        def _download_bytes(s3_key):
            try:
                data = self.download_bytes(s3_key)
                return s3_key, data, True, "Success"
            except Exception as e:
                return s3_key, None, False, str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {executor.submit(_download_bytes, s3_key): s3_key for s3_key in s3_keys}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_key)):
                s3_key, data, success, message = future.result()

                if success and data is not None:
                    results[s3_key] = data
                else:
                    results[s3_key] = None
                    logger.warning(f"Не удалось загрузить {s3_key}: {message}")

                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, len(s3_keys))

        successful_count = sum(1 for data in results.values() if data is not None)
        logger.info(f"Пакетная загрузка в память завершена: {successful_count}/{len(s3_keys)} успешно")
        return results

    def list_files(
        self, prefix: str = "", max_keys: int = 1000, file_extensions: Optional[List[str]] = None
    ) -> List[str]:
        """
        Получение списка файлов в S3 с фильтрацией.

        Получает список объектов в бакете S3 с возможностью фильтрации по префиксу
        и расширениям файлов. Поддерживает пагинацию для получения всех объектов
        в бакете или поддиректории.

        Параметры
        ----------
        prefix : str, optional
            Префикс для фильтрации объектов (по умолчанию "").
        max_keys : int, optional
            Максимальное количество ключей для возврата за один запрос (по умолчанию 1000).
        file_extensions : List[str], optional
            Список расширений файлов для фильтрации (по умолчанию None).

        Возвращает
        -------
        List[str]
            Список ключей объектов в S3, соответствующих критериям фильтрации.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> # Получить все файлы
        >>> all_files = s3_manager.list_files()
        >>> # Получить файлы с префиксом "images/"
        >>> image_files = s3_manager.list_files(prefix="images/")
        >>> # Получить только JPG и PNG файлы
        >>> image_files = s3_manager.list_files(
        ...     prefix="images/", 
        ...     file_extensions=[".jpg", ".png"]
        ... )
        """
        try:
            files = []
            continuation_token = None

            while True:
                list_args = {"Bucket": self.bucket_name, "Prefix": prefix, "MaxKeys": max_keys}

                if continuation_token:
                    list_args["ContinuationToken"] = continuation_token

                response = self.client.list_objects_v2(**list_args)

                if "Contents" in response:
                    for file in response["Contents"]:
                        key = file["Key"]
                        # Фильтрация по расширению
                        if file_extensions:
                            file_ext = Path(key).suffix.lower()
                            if file_ext not in [ext.lower() for ext in file_extensions]:
                                continue
                        files.append(key)

                if not response.get("IsTruncated"):
                    break

                continuation_token = response.get("NextContinuationToken")

            return files

        except Exception as e:
            logger.error(f"Ошибка при получении списка файлов: {e}")
            return []

    def get_file_info(self, s3_key: str) -> Optional[Dict]:
        """
        Получение информации о файле (метаданные, размер и т.д.).

        Получает метаданные объекта в S3, включая размер, дату последнего изменения,
        тип контента и пользовательские метаданные.

        Параметры
        ----------
        s3_key : str
            Ключ объекта в S3 для получения информации.

        Возвращает
        -------
        Optional[Dict]
            Словарь с информацией о файле, содержащий ключи:
            - "size": размер файла в байтах
            - "last_modified": дата последнего изменения
            - "content_type": тип контента файла
            - "metadata": пользовательские метаданные
            Возвращает None в случае ошибки.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> info = s3_manager.get_file_info("path/to/file.txt")
        >>> if info is not None:
        ...     print(f"Размер файла: {info['size']} bytes")
        ...     print(f"Тип контента: {info['content_type']}")
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "content_type": response.get("ContentType", ""),
                "metadata": response.get("Metadata", {}),
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о файле {s3_key}: {e}")
            return None

    def file_exists(self, s3_key: str) -> bool:
        """
        Проверка существования файла в S3.

        Проверяет, существует ли объект с заданным ключом в бакете S3.
        Использует head_object операцию, которая возвращает метаданные
        объекта без его загрузки.

        Параметры
        ----------
        s3_key : str
            Ключ объекта в S3 для проверки существования.

        Возвращает
        -------
        bool
            True, если файл существует, иначе False.

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> if s3_manager.file_exists("path/to/file.txt"):
        ...     print("Файл существует")
        ... else:
        ...     print("Файл не найден")
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
        except Exception as e:
            logger.error(f"Ошибка при проверке существования файла {s3_key}: {e}")
            return False

    # ========== СТАТИСТИКА И УТИЛИТЫ ==========

    def get_upload_stats(self) -> Dict:
        """
        Получение статистики загрузки.

        Возвращает копию словаря со статистикой операций загрузки файлов в S3.
        Статистика включает количество успешных и неудачных операций, а также
        общий объем загруженных данных.

        Возвращает
        -------
        Dict
            Словарь со статистикой загрузки, содержащий ключи:
            - "successful": количество успешных операций загрузки
            - "failed": количество неудачных операций загрузки
            - "total_size": общий объем загруженных данных в байтах

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> stats = s3_manager.get_upload_stats()
        >>> print(f"Успешных загрузок: {stats['successful']}")
        >>> print(f"Объем данных: {stats['total_size']} bytes")
        """
        return self._upload_stats.copy()

    def get_download_stats(self) -> Dict:
        """
        Получение статистики скачивания.

        Возвращает копию словаря со статистикой операций скачивания файлов из S3.
        Статистика включает количество успешных и неудачных операций, а также
        общий объем скачанных данных.

        Возвращает
        -------
        Dict
            Словарь со статистикой скачивания, содержащий ключи:
            - "successful": количество успешных операций скачивания
            - "failed": количество неудачных операций скачивания
            - "total_size": общий объем скачанных данных в байтах

        Примеры
        --------
        >>> s3_manager = S3Manager()
        >>> stats = s3_manager.get_download_stats()
        >>> print(f"Успешных скачиваний: {stats['successful']}")
        >>> print(f"Объем данных: {stats['total_size']} bytes")
        """
        return self._download_stats.copy()

    def reset_upload_stats(self):
        """
        Сброс статистики загрузки.

        Сбрасывает статистику операций загрузки файлов в S3, обнуляя счетчики
        успешных и неудачных операций, а также общий объем загруженных данных.
        Метод потокобезопасен и использует блокировку для защиты данных.
        """
        with self._upload_lock:
            self._upload_stats = {"successful": 0, "failed": 0, "total_size": 0}

    def reset_download_stats(self):
        """
        Сброс статистики скачивания.

        Сбрасывает статистику операций скачивания файлов из S3, обнуляя счетчики
        успешных и неудачных операций, а также общий объем скачанных данных.
        Метод потокобезопасен и использует блокировку для защиты данных.
        """
        with self._download_lock:
            self._download_stats = {"successful": 0, "failed": 0, "total_size": 0}

    def _get_content_type(self, file_path: str) -> str:
        """Определение Content-Type по расширению файла"""
        extension = Path(file_path).suffix.lower()

        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".json": "application/json",
            ".xml": "application/xml",
            ".zip": "application/zip",
        }

        return content_types.get(extension, "application/octet-stream")
