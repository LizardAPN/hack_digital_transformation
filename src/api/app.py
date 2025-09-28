фффimport os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

# Импортируем CV модель
from src.models.cv_model import create_cv_model, CVModel

# Инициализация FastAPI приложения
app = FastAPI(
    title="ML Model API", description="API для предоставления прогнозов модели машинного обучения", version="0.1.0"
)

# Глобальные переменные для модели и конфигурации
model = None
config = None
cv_model: Optional[CVModel] = None


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


def load_model_and_config():
    """Загрузка обученной модели и конфигурации"""
    global model, config

    # Загрузка конфигурации
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Загрузка модели
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print(f"Предупреждение: Файл модели {model_path} не найден")


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
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        # Преобразование признаков в DataFrame
        features_df = pd.DataFrame([request.features])

        # Сделать прогноз
        prediction = model.predict(features_df)[0]

        # Получить вероятность прогноза, если доступна
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_df)[0]
            probability = float(np.max(probabilities))

        return PredictionResponse(prediction=int(prediction), probability=probability)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка прогнозирования: {str(e)}")


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
                BuildingDetection(
                    bbox=building["bbox"],
                    confidence=building["confidence"],
                    area=building.get("area")
                ) for building in result["buildings"]
            ]
        )
        
        # Добавляем координаты, если есть
        if result.get("coordinates"):
            response.coordinates = CoordinateResult(
                lat=result["coordinates"]["lat"],
                lon=result["coordinates"]["lon"]
            )
        
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
                roi_name=ocr_data["roi_name"] if ocr_data["roi_name"] else None
            )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки изображения: {str(e)}")


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
