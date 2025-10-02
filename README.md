# Система обработки фотографий зданий

## Описание

Система предназначена для автоматического определения географических координат (WGS 84) и адресов зданий на фотоматериалах (Москва, МО) с использованием ИНС и компьютерного зрения.

## Основные функции

1. **Распознавание зданий** на фотографиях (выделение объектов bounding box)
2. **Определение координат** зданий на основе визуального поиска
3. **Привязка к адресам** с использованием внешних геокодеров (Яндекс, 2GIS, ФИАС и др.)
4. **Загрузка фотографий** по одной или пакетно через ZIP-архивы
5. **Экспорт результатов** в формате XLSX

## Архитектура

Система состоит из нескольких микросервисов:

1. **PhotoUploadService** - сервис загрузки фотографий
2. **Main API Service** - основной сервис обработки изображений
3. **AuthService** - сервис аутентификации
4. **Database** - база данных PostgreSQL

## Установка и запуск

### Требования

- Python 3.8+
- Docker и Docker Compose
- PostgreSQL
- Redis

### Локальный запуск

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Настройте переменные окружения в `.env` файле:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/building_detector
REDIS_URL=redis://localhost:6379/0
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_BUCKET_NAME=your_bucket_name
```

3. Запустите сервисы:
```bash
# Запуск базы данных и Redis
docker-compose up -d db redis

# Запуск основного API сервиса
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Запуск сервиса загрузки фотографий
uvicorn full-stack.PhotoUploadService.photo_upload:app --host 0.0.0.0 --port 8001
```

## Использование

### Аутентификация

Для работы с API необходимо авторизоваться. Все endpoints требуют наличия cookie с session_token.

### Загрузка фотографий

#### Одиночная загрузка
```http
POST /api/photo_upload
Content-Type: multipart/form-data
Cookie: session_token=your_token

file: image.jpg
```

#### Загрузка ZIP-архива
```http
POST /api/zip_upload
Content-Type: multipart/form-data
Cookie: session_token=your_token

file: photos.zip
```

### Получение результатов обработки

#### Получение списка фотографий пользователя
```http
GET /api/photos
Cookie: session_token=your_token
```

#### Экспорт результатов в Excel
```http
GET /export/results/xlsx
Cookie: session_token=your_token
```

### Поиск изображений

#### Поиск по координатам
```http
POST /search/by_coordinates
Content-Type: application/json

{
  "lat": 55.7558,
  "lon": 37.6173,
  "radius_km": 1.0
}
```

#### Поиск по адресу
```http
POST /search/by_address
Content-Type: application/json

{
  "address": "Москва, Красная площадь, 1"
}
```

## API Endpoints

### PhotoUploadService

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Проверка состояния сервиса |
| `/api/photo_upload` | POST | Загрузка одиночного изображения |
| `/api/zip_upload` | POST | Загрузка ZIP-архива с изображениями |
| `/api/photos` | GET | Получение списка загруженных фотографий |

### Main API Service

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Корневой эндпоинт с информацией об API |
| `/health` | GET | Проверка состояния сервиса |
| `/process_image` | POST | Синхронная обработка изображения |
| `/process_image_async` | POST | Асинхронная обработка изображения |
| `/process_images_batch` | POST | Пакетная асинхронная обработка изображений |
| `/task/{task_id}` | GET | Получение статуса задачи |
| `/tasks/request/{request_id}` | GET | Получение всех задач по request_id |
| `/results/latest` | GET | Получение последних результатов обработки |
| `/results/photo/{photo_id}` | GET | Получение результатов обработки для конкретного фото |
| `/search/by_coordinates` | POST | Поиск изображений по координатам |
| `/search/by_address` | POST | Поиск изображений по адресу |
| `/export/results/xlsx` | GET | Экспорт результатов обработки в формате XLSX |
| `/user/query_history` | POST | Сохранение истории запросов пользователя |
| `/user/query_history/{user_id}` | GET | Получение истории запросов пользователя |
| `/coordinates` | POST | Загрузка координат |
| `/coordinates/batch` | POST | Загрузка каталога координат |
| `/model/info` | GET | Получить информацию о модели |
| `/import/zip` | POST | Импорт данных zip архивом |
| `/download/image/{image_id}` | GET | Скачивание изображения |

## Разработка

### Структура проекта

```
├── src/                    # Основной исходный код
│   ├── api/               # API endpoints
│   ├── models/            # Модели машинного обучения
│   ├── geo/               # Геокодирование
│   ├── data/              # Работа с данными
│   ├── engine/            # Движок обработки
│   ├── tasks/             # Асинхронные задачи (Celery)
│   ├── utils/             # Вспомогательные функции
│   └── visualization/     # Визуализация
├── full-stack/            # Микросервисы
│   ├── PhotoUploadService/ # Сервис загрузки фотографий
│   ├── AuthService/       # Сервис аутентификации
│   ├── DB/               # Конфигурация базы данных
│   └── ngnix/            # Веб-интерфейс
├── notebooks/            # Jupyter notebooks для исследований
├── tests/                # Тесты
└── configs/              # Конфигурационные файлы
```

### Добавление новых функций

1. Создайте новую ветку для разработки
2. Реализуйте функционал в соответствующем модуле
3. Напишите тесты
4. Обновите документацию
5. Создайте pull request

## Лицензия

Проект распространяется под лицензией MIT.
