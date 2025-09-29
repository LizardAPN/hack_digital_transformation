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
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from fastapi import FastAPI, HTTPException
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


class TaskStatusResponse(BaseModel):
    """Модель ответа для статуса задачи"""

    task_id: str
    request_id: Optional[str] = None
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[dict] = None


def load_model_and_config():
    """Загрузка обученной модели и конфигурации"""
    global model, config

    # В текущей архитектуре не используется загрузка модели ML
    # Так как мы используем визуальный поиск по изображениям
    model = None
    config = None


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
        task = process_image_task.delay(request.image_path, request_id)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
