from fastapi import FastAPI, File, UploadFile, Cookie, HTTPException, Response, Depends
from pydantic import BaseModel
import psycopg2
import os
import uuid
import boto3
from botocore.exceptions import ClientError
from typing import List, Optional
import logging
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Переменные окружения
DATABASE_URL = os.getenv("DATABASE_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
S3_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
# URL API для обработки изображений
IMAGE_PROCESSING_API_URL = os.getenv("IMAGE_PROCESSING_API_URL", "http://fastapi:8000")

# Инициализация клиента S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=AWS_ENDPOINT_URL,
)


def trigger_image_processing(image_path: str, owner_id: int, request_id: Optional[str] = None, photo_id: Optional[int] = None, workspace_id: Optional[int] = None) -> bool:
    """
    Запуск обработки изображения путем вызова API обработки изображений
    
    Параметры
    ----------
    image_path : str
        Путь к изображению в S3
    owner_id : int
        Идентификатор владельца изображения
    request_id : Optional[str], optional
        Опциональный ID запроса для отслеживания, по умолчанию None
    photo_id : Optional[int], optional
        Опциональный ID фото для связи результатов, по умолчанию None
        
    Возвращает
    -------
    bool
        True если обработка была успешно запущена, False в противном случае
    """
    try:
        # Подготовка полезной нагрузки запроса
        payload = {
            "workspace_id": workspace_id,
            "owner_id": owner_id,
            "image_path": image_path,
        }
        
        # Добавляем request_id если он предоставлен
        if request_id:
            payload["request_id"] = request_id
            
        # Добавляем photo_id если он предоставлен
        if photo_id:
            payload["photo_id"] = str(photo_id)
            
        # Добавляем workspace_id если он предоставлен
        if workspace_id:
            payload["workspace_id"] = workspace_id
            
        # Вызов API обработки изображений
        response = requests.post(
            f"{IMAGE_PROCESSING_API_URL}/process_image_async",
            json=payload,
            timeout=30
        )
        
        # Проверка успешности запроса
        if response.status_code == 200:
            logger.info(f"Успешно запущена обработка изображения для {image_path}")
            return True
        else:
            logger.error(f"Не удалось запустить обработку изображения для {image_path}. Код состояния: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Ошибка при запуске обработки изображения для {image_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при запуске обработки изображения для {image_path}: {e}")
        return False

class PhotoResponse(BaseModel):
    """Модель ответа с информацией о фото"""
    id: int
    owner_id: int
    photo_url: str
    created_at: str

class UserPhotosResponse(BaseModel):
    """Модель ответа со списком фото пользователя"""
    photos: List[PhotoResponse]

def check_single_quote(*args):
    """
    Проверка наличия одинарных кавычек в аргументах для предотвращения SQL-инъекций
    
    Параметры
    ----------
    *args : tuple
        Аргументы для проверки
    """
    for arg in args:
        if isinstance(arg, str) and "'" in arg:
            raise HTTPException(status_code=400, detail="Недопустимый ввод: содержит одинарную кавычку")

def get_user_id_from_session(session_token: str) -> int:
    """
    Получение ID пользователя по токену сессии
    
    Параметры
    ----------
    session_token : str
        Токен сессии пользователя
        
    Возвращает
    -------
    int
        ID пользователя
        
    Выбрасывает
    ------
    HTTPException
        Если токен сессии отсутствует или недействителен
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="Токен сессии не предоставлен")
    
    check_single_quote(session_token)
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE session_token = %s", (session_token,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user:
            return user[0]  # Возвращаем ID пользователя
        else:
            raise HTTPException(status_code=401, detail="Недействительный токен сессии")
    except psycopg2.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")

@app.get("/")
def health():
    """Проверка состояния сервиса"""
    return {"Status": "ok"}

@app.post("/api/photo_upload")
async def upload_photo(
    file: UploadFile = File(...),
    session_token: str = Cookie(None),
    workspace_id: Optional[int] = None
):
    """
    Загрузка фото в S3 и сохранение метаданных в базу данных
    
    Параметры
    ----------
    file : UploadFile
        Файл изображения для загрузки
    session_token : str, optional
        Токен сессии пользователя из cookie
        
    Возвращает
    -------
    dict
        Информация о загруженном фото
    """
    # Аутентификация пользователя
    owner_id = get_user_id_from_session(session_token)
    
    # Проверка файла
    if not file:
        raise HTTPException(status_code=400, detail="Файл не предоставлен")
    
    # Генерация уникального имени файла
    file_extension = file.filename.split('.')[-1] if file.filename and '.' in file.filename else ''
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    
    try:
        # Загрузка файла в S3
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET_NAME,
            unique_filename,
            ExtraArgs={'ContentType': file.content_type or 'application/octet-stream'}
        )
        
        # Генерация URL файла в S3
        photo_url = unique_filename
        
        # Сохранение метаданных в базу данных
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Формируем SQL-запрос в зависимости от наличия workspace_id
        if workspace_id:
            cur.execute(
                "INSERT INTO user_photos (owner_id, photo_url, workspace_id) VALUES (%s, %s, %s) RETURNING id, created_at",
                (owner_id, photo_url, workspace_id)
            )
        else:
            cur.execute(
                "INSERT INTO user_photos (owner_id, photo_url) VALUES (%s, %s) RETURNING id, created_at",
                (owner_id, photo_url)
            )
        
        result = cur.fetchone()
        if result is None:
            raise HTTPException(status_code=500, detail="Не удалось вставить фото в базу данных")
        photo_id = result[0]
        created_at = result[1]
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Запуск обработки изображения
        processing_triggered = trigger_image_processing(photo_url, owner_id, str(photo_id), photo_id, workspace_id)
        
        return {
            "id": photo_id,
            "photo_url": photo_url,
            "created_at": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at),
            "processing_triggered": processing_triggered
        }
        
    except ClientError as e:
        logger.error(f"Ошибка загрузки в S3: {e}")
        raise HTTPException(status_code=500, detail="Не удалось загрузить файл в S3")
    except psycopg2.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.get("/api/photos", response_model=UserPhotosResponse)
def get_user_photos(session_token: str = Cookie(None), page: int = 1, limit: int = 10, workspace_id: Optional[int] = None):
    """
    Получение фотографий аутентифицированного пользователя с пагинацией
    
    Параметры
    ----------
    session_token : str, optional
        Токен сессии пользователя из cookie
    page : int, optional
        Номер страницы (по умолчанию 1)
    limit : int, optional
        Количество фотографий на странице (по умолчанию 10)
        
    Возвращает
    -------
    UserPhotosResponse
        Список фотографий пользователя для указанной страницы
    """
    # Аутентификация пользователя
    owner_id = get_user_id_from_session(session_token)
    
    # Валидация параметров пагинации
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:
        limit = 10
    
    try:
        # Получение фотографий из базы данных с учетом пагинации
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Вычисляем смещение для пагинации
        offset = (page - 1) * limit
        
        # Формируем SQL-запрос в зависимости от наличия workspace_id
        if workspace_id:
            cur.execute(
                "SELECT id, owner_id, photo_url, created_at FROM user_photos WHERE owner_id = %s AND workspace_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (owner_id, workspace_id, limit, offset)
            )
        else:
            cur.execute(
                "SELECT id, owner_id, photo_url, created_at FROM user_photos WHERE owner_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (owner_id, limit, offset)
            )
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Формирование ответа
        photos = []
        for row in rows:
            photos.append({
                "id": row[0],
                "owner_id": row[1],
                "photo_url": f"https://storage.yandexcloud.net/{S3_BUCKET_NAME}/{row[2]}",
                "created_at": row[3].isoformat() if hasattr(row[3], 'isoformat') else str(row[3])
            })
        
        return {"photos": photos}
        
    except psycopg2.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/zip_upload")
async def upload_zip(
    file: UploadFile = File(...),
    session_token: str = Cookie(None),
    workspace_id: Optional[int] = None
):
    """
    Загрузка ZIP-файла с фотографиями в S3 и сохранение метаданных в базу данных
    
    Параметры
    ----------
    file : UploadFile
        ZIP-файл с изображениями для загрузки
    session_token : str, optional
        Токен сессии пользователя из cookie
        
    Возвращает
    -------
    dict
        Информация о загруженных файлах и результатах обработки
        
    Выбрасывает
    ------
    HTTPException
        При ошибке загрузки, обработки или некорректном формате файла
    """
    # Аутентификация пользователя
    owner_id = get_user_id_from_session(session_token)
    
    # Проверка файла
    if not file:
        raise HTTPException(status_code=400, detail="Файл не предоставлен")
    
    # Проверка, что файл является ZIP-архивом
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Разрешены только ZIP-файлы")
    
    try:
        # Создание временной директории для распаковки
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Временное сохранение загруженного ZIP-файла
            zip_path = temp_path / file.filename
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Распаковка ZIP-файла
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # Обработка распакованных файлов
            processed_files = []
            failed_files = []
            
            # Поддерживаемые расширения изображений
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            # Итерация по распакованным файлам
            for root, dirs, files in os.walk(temp_path):
                for filename in files:
                    file_path = Path(root) / filename
                    extension = file_path.suffix.lower()
                    
                    # Обрабатываем только файлы изображений
                    if extension in image_extensions:
                        try:
                            # Генерация уникального имени файла
                            unique_filename = f"{uuid.uuid4()}{extension}"
                            
                            # Загрузка файла в S3
                            with open(file_path, 'rb') as img_file:
                                s3_client.upload_fileobj(
                                    img_file,
                                    S3_BUCKET_NAME,
                                    unique_filename,
                                    ExtraArgs={'ContentType': f'image/{extension[1:]}'}
                                )
                            
                            # Сохранение метаданных в базу данных
                            conn = psycopg2.connect(DATABASE_URL)
                            cur = conn.cursor()
                            
                            # Формируем SQL-запрос в зависимости от наличия workspace_id
                            if workspace_id:
                                cur.execute(
                                    "INSERT INTO user_photos (owner_id, photo_url, workspace_id) VALUES (%s, %s, %s) RETURNING id, created_at",
                                    (owner_id, unique_filename, workspace_id)
                                )
                            else:
                                cur.execute(
                                    "INSERT INTO user_photos (owner_id, photo_url) VALUES (%s, %s) RETURNING id, created_at",
                                    (owner_id, unique_filename)
                                )
                            
                            result = cur.fetchone()
                            if result is None:
                                raise HTTPException(status_code=500, detail="Не удалось вставить фото в базу данных")
                            photo_id = result[0]
                            created_at = result[1]
                            
                            conn.commit()
                            cur.close()
                            conn.close()
                            
                            # Запуск обработки изображения
                            processing_triggered = trigger_image_processing(unique_filename, owner_id, str(photo_id), photo_id, workspace_id)
                            
                            processed_files.append({
                                "filename": filename,
                                "id": photo_id,
                                "photo_url": unique_filename,
                                "created_at": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at),
                                "processing_triggered": processing_triggered
                            })
                            
                        except Exception as e:
                            logger.error(f"Ошибка обработки файла {filename}: {e}")
                            failed_files.append({
                                "filename": filename,
                                "error": str(e)
                            })
            
            return {
                "message": f"ZIP-файл обработан. Успешно загружено {len(processed_files)} файлов, {len(failed_files)} файлов не удалось обработать.",
                "processed_files": processed_files,
                "failed_files": failed_files
            }
            
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Некорректный ZIP-файл")
    except ClientError as e:
        logger.error(f"Ошибка загрузки в S3: {e}")
        raise HTTPException(status_code=500, detail="Не удалось загрузить файлы в S3")
    except psycopg2.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
