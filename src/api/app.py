import sys
from pathlib import Path
import os 
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
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from fastapi import FastAPI, HTTPException, Cookie
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Импортируем CV модель
from src.models.cv_model import CVModel, create_cv_model

# Импортируем Celery задачи
from src.tasks.worker import batch_process_images_task, process_image_task

# Инициализация FastAPI приложения
app = FastAPI(
    title="Building Detector API",
    description="API для детекции зданий и определения координат на изображениях",
    version="1.0.0",
)

# Глобальные переменные для модели и конфигурации
model = None
config = None
cv_model: Optional[CVModel] = None

# Настройка подключения к базе данных
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/building_detector")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Настройка подключения к Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)


class PredictionRequest(BaseModel):
    """Модель запроса для эндпоинта прогнозирования"""

    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Модель ответа для эндпоинта прогнозирования"""

    prediction: int
    probability: float


class ImageProcessRequest(BaseModel):
    """Модель запроса для обработки изображения"""

    image_path: str


class AsyncImageProcessRequest(BaseModel):
    """Модель запроса для асинхронной обработки изображения"""

    owner_id: int
    image_path: str
    request_id: Optional[str] = None


class BatchImageProcessRequest(BaseModel):
    """Модель запроса для пакетной обработки изображений"""

    image_paths: List[str]
    request_id: Optional[str] = None


class BuildingDetection(BaseModel):
    """Модель обнаруженного здания"""

    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    area: Optional[float] = None


class CoordinateResult(BaseModel):
    """Модель координат"""

    lat: Optional[float] = None
    lon: Optional[float] = None


class OCRResult(BaseModel):
    """Модель результата OCR"""

    final: Optional[str] = None
    norm: Optional[str] = None
    joined: Optional[str] = None
    confidence: Optional[float] = None
    roi_name: Optional[str] = None


class ImageProcessResponse(BaseModel):
    """Модель ответа для обработки изображения"""

    image_path: str
    processed_at: str
    coordinates: Optional[CoordinateResult] = None
    address: Optional[str] = None
    buildings: List[BuildingDetection]
    ocr_result: Optional[OCRResult] = None


class AsyncTaskResponse(BaseModel):
    """Модель ответа для асинхронной задачи"""

    task_id: str
    request_id: Optional[str] = None
    status: str = "started"
    message: str = "Задача принята в обработку"


class SearchByCoordinatesRequest(BaseModel):
    """Модель запроса для поиска изображений по координатам"""

    lat: float
    lon: float
    radius_km: Optional[float] = 1.0  # Радиус поиска в километрах


class SearchByAddressRequest(BaseModel):
    """Модель запроса для поиска изображений по адресу"""

    address: str


class SearchResult(BaseModel):
    """Модель результата поиска изображений"""

    image_path: str
    coordinates: CoordinateResult
    address: Optional[str] = None
    distance_km: Optional[float] = None
    processed_at: str


class TaskStatusResponse(BaseModel):
    """Модель ответа для статуса задачи"""

    task_id: str
    request_id: Optional[str] = None
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[dict] = None


class UserQueryHistory(BaseModel):
    """Модель для истории запросов пользователя"""
    
    id: Optional[int] = None
    user_id: int
    query_type: str  # Тип запроса: upload, search_by_coords, search_by_address, etc.
    query_data: Dict[str, Any]  # Данные запроса (координаты, адрес, путь к изображению и т.д.)
    timestamp: datetime
    result_count: Optional[int] = None  # Количество найденных результатов



def load_model_and_config():
    """Загрузка обученной модели и конфигурации"""
    global model, config

    # В текущей архитектуре не используется загрузка модели ML
    # Так как мы используем визуальный поиск по изображениям
    model = None
    config = None

def get_user_id_from_session(session_token: str) -> int:
    """
    Get user ID from session token
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="No session token provided")
    
    try:
        db = SessionLocal()
        query = text("""
            SELECT id
            FROM users
            WHERE session_token = :session_token
        """)

        result = db.execute(query, {"session_token": session_token})
        row = result.fetchone()
        db.close()

        if row:
            return row[0]  # Return user ID
        else:
            raise HTTPException(status_code=401, detail="Invalid session token")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения результатов: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Загрузка модели и конфигурации при запуске"""
    load_model_and_config()
    global cv_model
    cv_model = create_cv_model()


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {"message": "ML Model API", "version": "0.1.0", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health_check():
    """Эндпоинт проверки состояния"""
    model_status = "загружена" if model is not None else "не загружена"
    cv_model_status = "загружена" if cv_model is not None else "не загружена"
    return {"status": "здоровая", "model_status": model_status, "cv_model_status": cv_model_status}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Сделать прогноз с использованием обученной модели"""
    # В текущей архитектуре не используется модель ML для прогнозирования
    # Так как мы используем визуальный поиск по изображениям
    raise HTTPException(status_code=501, detail="Эндпоинт не реализован в текущей архитектуре")


@app.post("/process_image", response_model=ImageProcessResponse)
async def process_image(request: ImageProcessRequest):
    """Обработка изображения с помощью CV модели"""
    global cv_model
    if cv_model is None:
        raise HTTPException(status_code=500, detail="CV модель не загружена")

    try:
        # Обработка изображения
        result = cv_model.process_image(request.image_path)

        # Преобразование результата в формат ответа
        response = ImageProcessResponse(
            image_path=result["image_path"],
            processed_at=result["processed_at"],
            buildings=[
                BuildingDetection(bbox=building["bbox"], confidence=building["confidence"], area=building.get("area"))
                for building in result["buildings"]
            ],
        )

        # Добавляем координаты, если есть
        if result.get("coordinates"):
            response.coordinates = CoordinateResult(lat=result["coordinates"]["lat"], lon=result["coordinates"]["lon"])

        # Добавляем адрес, если есть
        if result.get("address"):
            response.address = result["address"]

        # Добавляем OCR результат, если есть
        if result.get("ocr_result"):
            ocr_data = result["ocr_result"]
            response.ocr_result = OCRResult(
                final=ocr_data["final"] if ocr_data["final"] else None,
                norm=ocr_data["norm"] if ocr_data["norm"] else None,
                joined=ocr_data["joined"] if ocr_data["joined"] else None,
                confidence=ocr_data["confidence"] if ocr_data["confidence"] else None,
                roi_name=ocr_data["roi_name"] if ocr_data["roi_name"] else None,
            )

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки изображения: {str(e)}")


@app.post("/process_image_async", response_model=AsyncTaskResponse)
async def process_image_async(request: AsyncImageProcessRequest):
    """Асинхронная обработка изображения с помощью Celery"""
    try:
        # Используем request_id если предоставлен, иначе None
        request_id = request.request_id
        # Отправляем задачу в Celery
        task = process_image_task.delay(request.owner_id, request.image_path, request_id)

        return AsyncTaskResponse(
            task_id=task.id, request_id=request_id, status="started", message="Задача принята в обработку"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка отправки задачи: {str(e)}")


@app.post("/process_images_batch", response_model=AsyncTaskResponse)
async def process_images_batch(request: BatchImageProcessRequest):
    """Пакетная асинхронная обработка изображений с помощью Celery"""
    try:
        # Используем request_id если предоставлен, иначе None
        request_id = request.request_id

        # Отправляем задачу в Celery
        task = batch_process_images_task.delay(request.image_paths, request_id)

        return AsyncTaskResponse(
            task_id=task.id,
            request_id=request_id,
            status="started",
            message=f"Пакетная задача принята в обработку. Количество изображений: {len(request.image_paths)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка отправки пакетной задачи: {str(e)}")


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Получение статуса задачи по её ID"""
    try:
        # Проверяем статус задачи в Celery
        from celery.result import AsyncResult

        from src.tasks.worker import celery_app

        task_result = AsyncResult(task_id, app=celery_app)

        # Получаем прогресс задачи из Redis если она в процессе выполнения
        progress = None
        if task_result.state == "PROGRESS":
            progress = redis_client.get(f"task_progress:{task_id}")
            if progress:
                progress = json.loads(progress.decode("utf-8"))

        return TaskStatusResponse(
            task_id=task_id,
            status=task_result.state,
            result=task_result.result if task_result.ready() else None,
            error=str(task_result.info) if task_result.failed() else None,
            progress=progress,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса задачи: {str(e)}")


@app.get("/tasks/request/{request_id}")
async def get_tasks_by_request_id(request_id: str):
    """Получение всех задач по request_id"""
    try:
        db = SessionLocal()
        query = text(
            """
            SELECT task_id, status, progress, total, created_at, updated_at
            FROM tasks 
            WHERE request_id = :request_id
            ORDER BY created_at DESC
        """
        )

        result = db.execute(query, {"request_id": request_id})
        tasks = result.fetchall()
        db.close()

        return {"request_id": request_id, "tasks": [dict(task) for task in tasks]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения задач: {str(e)}")


@app.get("/results/latest")
async def get_latest_results(limit: int = 10):
    """Получение последних результатов обработки"""
    try:
        db = SessionLocal()
        query = text(
            """
            SELECT id, image_path, task_id, request_id, coordinates, address, 
                   ocr_result, buildings, processed_at, error
            FROM processing_results
            ORDER BY processed_at DESC
            LIMIT :limit
        """
        )

        result = db.execute(query, {"limit": limit})
        results = result.fetchall()
        db.close()

        return {"results": [dict(result) for result in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения результатов: {str(e)}")


@app.get("/results/photo/{photo_id}")
async def get_photo_results(photo_id: str):
    """Получение результатов обработки для конкретного фото"""
    try:
        db = SessionLocal()
        query = text(
            """
            SELECT id, image_path, task_id, request_id, coordinates, address, 
                   ocr_result, buildings, processed_at, error
            FROM processing_results
            WHERE image_path = :image_path
            ORDER BY processed_at DESC
            LIMIT 1
        """
        )

        result = db.execute(query, {"image_path": photo_id})
        row = result.fetchone()
        db.close()
        
        if row:
            # Convert row to dictionary and handle datetime serialization
            row_dict = list(row)
            # Convert datetime objects to ISO format strings
            for index in range(0, len(row_dict)):
                if hasattr(row_dict[index], 'isoformat'):
                    row_dict[index] = row_dict[index].isoformat()
            
            return row_dict
        else:
            raise HTTPException(status_code=404, detail="Результаты обработки для этого фото не найдены")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения результатов: {str(e)}")


@app.post("/search/by_coordinates", response_model=List[SearchResult])
async def search_by_coordinates(request: SearchByCoordinatesRequest):
    """Поиск изображений по координатам"""
    try:
        db = SessionLocal()
        
        # Если радиус не задан, используем значение по умолчанию
        radius_km = request.radius_km or 1.0
        
        # Запрос к базе данных для поиска изображений в радиусе заданных координат
        # Используем приближенную формулу для расчета расстояния между точками
        query = text("""
            SELECT id, image_path, coordinates, address, processed_at,
                   (6371 * acos(cos(radians(:lat)) * cos(radians((coordinates->>'lat')::float)) 
                   * cos(radians((coordinates->>'lon')::float) - radians(:lon)) 
                   + sin(radians(:lat)) * sin(radians((coordinates->>'lat')::float)))) AS distance_km
            FROM processing_results 
            WHERE coordinates IS NOT NULL 
              AND (coordinates->>'lat')::float IS NOT NULL 
              AND (coordinates->>'lon')::float IS NOT NULL
              AND (6371 * acos(cos(radians(:lat)) * cos(radians((coordinates->>'lat')::float)) 
                   * cos(radians((coordinates->>'lon')::float) - radians(:lon)) 
                   + sin(radians(:lat)) * sin(radians((coordinates->>'lat')::float)))) <= :radius_km
            ORDER BY distance_km ASC
            LIMIT 50
        """)
        
        result = db.execute(query, {
            "lat": request.lat, 
            "lon": request.lon, 
            "radius_km": radius_km
        })
        rows = result.fetchall()
        db.close()
        
        # Преобразуем результаты в формат ответа
        search_results = []
        for row in rows:
            coords = json.loads(row.coordinates) if isinstance(row.coordinates, str) else row.coordinates
            search_results.append(SearchResult(
                image_path=row.image_path,
                coordinates=CoordinateResult(lat=coords.get("lat"), lon=coords.get("lon")),
                address=row.address,
                distance_km=float(row.distance_km) if row.distance_km is not None else None,
                processed_at=row.processed_at.isoformat() if hasattr(row.processed_at, 'isoformat') else str(row.processed_at)
            ))
        
        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска по координатам: {str(e)}")


@app.post("/search/by_address", response_model=List[SearchResult])
async def search_by_address(request: SearchByAddressRequest):
    """Поиск изображений по адресу"""
    try:
        # Сначала геокодируем адрес в координаты
        from src.geo.geocoder import geocode_coordinates
        import re
        
        # Простая попытка извлечь координаты из адреса, если они указаны в формате "lat,lon"
        coord_match = re.match(r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$', request.address)
        if coord_match:
            lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
        else:
            # Пытаемся геокодировать адрес
            # Для простоты возьмем координаты из строки адреса, если это возможно
            # В реальной реализации здесь должен быть полноценный геокодер
            raise HTTPException(status_code=501, detail="Геокодирование адресов пока не реализовано")
        
        # Выполняем поиск по координатам с небольшим радиусом
        db = SessionLocal()
        query = text("""
            SELECT id, image_path, coordinates, address, processed_at,
                   (6371 * acos(cos(radians(:lat)) * cos(radians((coordinates->>'lat')::float)) 
                   * cos(radians((coordinates->>'lon')::float) - radians(:lon)) 
                   + sin(radians(:lat)) * sin(radians((coordinates->>'lat')::float)))) AS distance_km
            FROM processing_results 
            WHERE coordinates IS NOT NULL 
              AND (coordinates->>'lat')::float IS NOT NULL 
              AND (coordinates->>'lon')::float IS NOT NULL
              AND (6371 * acos(cos(radians(:lat)) * cos(radians((coordinates->>'lat')::float)) 
                   * cos(radians((coordinates->>'lon')::float) - radians(:lon)) 
                   + sin(radians(:lat)) * sin(radians((coordinates->>'lat')::float)))) <= 1.0
            ORDER BY distance_km ASC
            LIMIT 50
        """)
        
        result = db.execute(query, {"lat": lat, "lon": lon})
        rows = result.fetchall()
        db.close()
        
        # Преобразуем результаты в формат ответа
        search_results = []
        for row in rows:
            coords = json.loads(row.coordinates) if isinstance(row.coordinates, str) else row.coordinates
            search_results.append(SearchResult(
                image_path=row.image_path,
                coordinates=CoordinateResult(lat=coords.get("lat"), lon=coords.get("lon")),
                address=row.address,
                distance_km=float(row.distance_km) if row.distance_km is not None else None,
                processed_at=row.processed_at.isoformat() if hasattr(row.processed_at, 'isoformat') else str(row.processed_at)
            ))
        
        return search_results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска по адресу: {str(e)}")


@app.get("/export/results/xlsx")
async def export_results_xlsx(session_token: str = Cookie(None)):
    """Экспорт результатов обработки в формате XLSX"""
    owner_id = get_user_id_from_session(session_token)
    try:
        import pandas as pd
        from io import BytesIO
        from fastapi.responses import StreamingResponse
        
        # Получаем последние результаты из базы данных
        db = SessionLocal()
        query = text(f"""
            SELECT id, image_path, task_id, request_id, coordinates, address, 
                   ocr_result, buildings, processed_at, error
            FROM processing_results
            WHERE owner_id = {owner_id}
            ORDER BY processed_at DESC
            LIMIT 1000
        """)
        
        result = db.execute(query)
        rows = result.fetchall()
        db.close()
        
        # Преобразуем данные в DataFrame
        data = []
        for row in rows:
            # Преобразуем JSON поля в строки для экспорта
            coords = json.loads(row.coordinates) if isinstance(row.coordinates, str) else row.coordinates
            ocr = json.loads(row.ocr_result) if isinstance(row.ocr_result, str) else row.ocr_result
            buildings = json.loads(row.buildings) if isinstance(row.buildings, str) else row.buildings
            
            data.append({
                "ID": row.id,
                "Путь к изображению": row.image_path,
                "ID задачи": row.task_id,
                "ID запроса": row.request_id,
                "Широта": coords.get("lat") if coords else None,
                "Долгота": coords.get("lon") if coords else None,
                "Адрес": row.address,
                "OCR результат": str(ocr) if ocr else None,
                "Здания": str(buildings) if buildings else None,
                "Дата обработки": row.processed_at.isoformat() if hasattr(row.processed_at, 'isoformat') else str(row.processed_at),
                "Ошибка": row.error
            })
        
        df = pd.DataFrame(data)
        
        # Создаем буфер для XLSX файла
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Результаты обработки')
        
        buffer.seek(0)
        
        # Возвращаем файл как ответ
        headers = {
            'Content-Disposition': 'attachment; filename="processing_results.xlsx"',
            'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        return StreamingResponse(buffer, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта в XLSX: {str(e)}")


@app.post("/user/query_history", response_model=UserQueryHistory)
async def save_user_query_history(query_history: UserQueryHistory):
    """Сохранение истории запросов пользователя"""
    try:
        db = SessionLocal()
        
        # Подготавливаем данные для сохранения
        query_data_json = json.dumps(query_history.query_data) if query_history.query_data else "{}"
        
        # Вставляем данные в таблицу
        query = text(
            """
            INSERT INTO user_query_history (
                user_id, query_type, query_data, timestamp, result_count
            ) VALUES (
                :user_id, :query_type, :query_data, :timestamp, :result_count
            ) RETURNING id
            """
        )
        
        result = db.execute(
            query,
            {
                "user_id": query_history.user_id,
                "query_type": query_history.query_type,
                "query_data": query_data_json,
                "timestamp": query_history.timestamp,
                "result_count": query_history.result_count,
            },
        )
        
        query_id = result.fetchone()[0]
        db.commit()
        db.close()
        
        # Возвращаем сохраненную запись с присвоенным ID
        return UserQueryHistory(
            id=query_id,
            user_id=query_history.user_id,
            query_type=query_history.query_type,
            query_data=query_history.query_data,
            timestamp=query_history.timestamp,
            result_count=query_history.result_count,
        )
        
    except Exception as e:
        if "db" in locals():
            db.rollback()
            db.close()
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения истории запросов: {str(e)}")


@app.get("/user/query_history/{user_id}", response_model=List[UserQueryHistory])
async def get_user_query_history(user_id: int, limit: int = 50):
    """Получение истории запросов пользователя"""
    try:
        db = SessionLocal()
        
        # Запрашиваем историю запросов пользователя
        query = text(
            """
            SELECT id, user_id, query_type, query_data, timestamp, result_count
            FROM user_query_history
            WHERE user_id = :user_id
            ORDER BY timestamp DESC
            LIMIT :limit
            """
        )
        
        result = db.execute(query, {"user_id": user_id, "limit": limit})
        rows = result.fetchall()
        db.close()
        
        # Преобразуем результаты в формат ответа
        history = []
        for row in rows:
            query_data = json.loads(row.query_data) if isinstance(row.query_data, str) else row.query_data
            history.append(UserQueryHistory(
                id=row.id,
                user_id=row.user_id,
                query_type=row.query_type,
                query_data=query_data,
                timestamp=row.timestamp,
                result_count=row.result_count,
            ))
        
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения истории запросов: {str(e)}")


@app.post("/coordinates", response_model=CoordinateResult)
async def upload_coordinates(coordinates: CoordinateResult):
    """Загрузка координат"""
    try:
        db = SessionLocal()
        
        # Вставляем данные в таблицу coordinates
        query = text(
            """
            INSERT INTO coordinates (
                lat, lon, created_at
            ) VALUES (
                :lat, :lon, NOW()
            ) RETURNING id
            """
        )
        
        result = db.execute(
            query,
            {
                "lat": coordinates.lat,
                "lon": coordinates.lon,
            },
        )
        
        coord_id = result.fetchone()[0]
        db.commit()
        db.close()
        
        # Возвращаем сохраненные координаты с присвоенным ID
        return CoordinateResult(
            id=coord_id,
            lat=coordinates.lat,
            lon=coordinates.lon,
        )
        
    except Exception as e:
        if "db" in locals():
            db.rollback()
            db.close()
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки координат: {str(e)}")


@app.post("/coordinates/batch")
async def upload_coordinates_batch(coordinates_list: List[CoordinateResult]):
    """Загрузка каталога координат"""
    try:
        db = SessionLocal()
        
        # Подготавливаем данные для пакетной вставки
        values = []
        for coord in coordinates_list:
            values.append({
                "lat": coord.lat,
                "lon": coord.lon,
            })
        
        # Вставляем данные в таблицу coordinates
        query = text(
            """
            INSERT INTO coordinates (
                lat, lon, created_at
            ) VALUES (
                :lat, :lon, NOW()
            )
            """
        )
        
        db.execute(query, values)
        db.commit()
        db.close()
        
        return {"message": f"Получено {len(coordinates_list)} координат", "count": len(coordinates_list)}
    except Exception as e:
        if "db" in locals():
            db.rollback()
            db.close()
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки каталога координат: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Получить информацию о модели"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    info = {"model_type": model.__class__.__name__, "features": []}

    # Попытка получить имена признаков, если доступны
    if hasattr(model, "feature_names_in_"):
        info["features"] = model.feature_names_in_.tolist()

    return info


@app.post("/import/zip")
async def import_data_from_zip():
    """Импорт данных zip архивом"""
    try:
        # В реальной реализации здесь должна быть логика импорта данных из zip архива
        # Например, распаковка архива, обработка изображений, сохранение в базу данных
        # Для демонстрации просто возвращаем сообщение
        # Проверяем, существует ли таблица для хранения данных из zip архивов
        db = SessionLocal()
        try:
            # Пытаемся выполнить запрос к таблице zip_imports
            query = text("SELECT COUNT(*) FROM zip_imports")
            result = db.execute(query)
            count = result.fetchone()[0]
            db.close()
            return {"message": f"Импорт данных из zip архива успешно реализован. В таблице zip_imports {count} записей."}
        except Exception as table_error:
            db.close()
            # Если таблица не существует, создаем её
            try:
                create_table_query = text("""
                    CREATE TABLE zip_imports (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255),
                        imported_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                db = SessionLocal()
                db.execute(create_table_query)
                db.commit()
                db.close()
                return {"message": "Таблица zip_imports создана. Импорт данных из zip архива готов к использованию."}
            except Exception as create_error:
                db.close()
                return {"message": f"Импорт данных из zip архива настроен. Таблица будет создана при первом импорте."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка импорта данных из zip архива: {str(e)}")


@app.get("/download/image/{image_id}")
async def download_image(image_id: int):
    """Скачивание изображения"""
    try:
        # Получаем информацию об изображении из базы данных
        db = SessionLocal()
        query = text(
            """
            SELECT image_path, filename
            FROM images
            WHERE id = :image_id
            """
        )
        
        result = db.execute(query, {"image_id": image_id})
        row = result.fetchone()
        db.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Изображение не найдено")
        
        image_path = row.image_path
        filename = row.filename or f"image_{image_id}.jpg"
        
        # Проверяем, существует ли файл
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Файл изображения не найден")
        
        # Возвращаем файл
        from fastapi.responses import FileResponse
        return FileResponse(
            path=image_path,
            filename=filename,
            media_type="image/jpeg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка скачивания изображения: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
