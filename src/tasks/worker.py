import json
import logging
import os
from datetime import datetime

from celery import Celery
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Импортируем модели и компоненты
from src.models.cv_model import CVModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание Celery приложения
celery_app = Celery("building_detector")
celery_app.conf.update(
    broker_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    result_backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Настройки производительности
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", "4")),
)

# Настройка подключения к базе данных
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/building_detector")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Глобальная модель CV
cv_model = None


def get_cv_model():
    """Получение глобального экземпляра CV модели"""
    global cv_model
    if cv_model is None:
        cv_model = CVModel()
    return cv_model


@celery_app.task(bind=True)
def process_image_task(self, image_path: str, request_id: str = None) -> dict:
    """
    Асинхронная задача для обработки изображения

    Args:
        image_path: Путь к изображению
        request_id: Идентификатор запроса (для отслеживания)

    Returns:
        Результат обработки изображения
    """
    try:
        logger.info(f"Начало обработки изображения: {image_path}")

        # Получаем модель CV
        model = get_cv_model()

        # Обрабатываем изображение
        result = model.process_image(image_path)

        # Добавляем информацию о задаче
        result["task_id"] = self.request.id
        result["request_id"] = request_id
        result["processed_at"] = datetime.now().isoformat()

        # Сохраняем результат в базу данных
        save_result_to_db(result)

        logger.info(f"Обработка изображения завершена: {image_path}")
        return result

    except Exception as e:
        logger.error(f"Ошибка обработки изображения {image_path}: {e}")
        # Сохраняем информацию об ошибке
        error_result = {
            "image_path": image_path,
            "task_id": self.request.id,
            "request_id": request_id,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
        }
        save_result_to_db(error_result)
        raise


def save_result_to_db(result: dict):
    """
    Сохранение результата обработки в базу данных

    Args:
        result: Результат обработки изображения
    """
    try:
        db = SessionLocal()

        # Подготавливаем данные для сохранения
        image_path = result.get("image_path", "")
        task_id = result.get("task_id", "")
        request_id = result.get("request_id", "")
        coordinates = result.get("coordinates", {})
        address = result.get("address", "")
        ocr_result = result.get("ocr_result", {})
        buildings = result.get("buildings", [])
        processed_at = result.get("processed_at", "")
        error = result.get("error", "")

        # Преобразуем сложные объекты в JSON
        coordinates_json = json.dumps(coordinates) if coordinates else "{}"
        ocr_result_json = json.dumps(ocr_result) if ocr_result else "{}"
        buildings_json = json.dumps(buildings) if buildings else "[]"

        # Вставляем данные в таблицу
        query = text(
            """
            INSERT INTO processing_results (
                image_path, task_id, request_id, coordinates, address, 
                ocr_result, buildings, processed_at, error
            ) VALUES (
                :image_path, :task_id, :request_id, :coordinates, :address,
                :ocr_result, :buildings, :processed_at, :error
            )
        """
        )

        db.execute(
            query,
            {
                "image_path": image_path,
                "task_id": task_id,
                "request_id": request_id,
                "coordinates": coordinates_json,
                "address": address,
                "ocr_result": ocr_result_json,
                "buildings": buildings_json,
                "processed_at": processed_at,
                "error": error,
            },
        )

        db.commit()
        db.close()

        logger.info(f"Результат сохранен в БД для задачи {task_id}")

    except Exception as e:
        logger.error(f"Ошибка сохранения результата в БД: {e}")
        if "db" in locals():
            db.rollback()
            db.close()


@celery_app.task(bind=True)
def batch_process_images_task(self, image_paths: list, request_id: str = None) -> dict:
    """
    Асинхронная задача для пакетной обработки изображений

    Args:
        image_paths: Список путей к изображениям
        request_id: Идентификатор запроса (для отслеживания)

    Returns:
        Сводный результат обработки
    """
    try:
        logger.info(f"Начало пакетной обработки {len(image_paths)} изображений")

        results = []
        errors = []

        # Обрабатываем изображения по одному
        for i, image_path in enumerate(image_paths):
            try:
                # Обновляем прогресс задачи
                self.update_state(state="PROGRESS", meta={"current": i, "total": len(image_paths)})

                # Обрабатываем изображение
                result = process_image_task(image_path, request_id)
                results.append(result)

            except Exception as e:
                logger.error(f"Ошибка обработки изображения {image_path}: {e}")
                errors.append({"image_path": image_path, "error": str(e)})

        # Формируем сводный результат
        summary = {
            "task_id": self.request.id,
            "request_id": request_id,
            "total_processed": len(results),
            "total_errors": len(errors),
            "results": results,
            "errors": errors,
            "processed_at": datetime.now().isoformat(),
        }

        logger.info(f"Пакетная обработка завершена. Обработано: {len(results)}, Ошибок: {len(errors)}")
        return summary

    except Exception as e:
        logger.error(f"Ошибка пакетной обработки: {e}")
        raise


if __name__ == "__main__":
    celery_app.start()
