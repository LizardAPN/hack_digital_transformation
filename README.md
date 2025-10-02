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
5. **Celery Worker** - асинхронные задачи обработки изображений

### Диаграмма архитектуры

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Nginx Proxy    │    │   PostgreSQL    │
│  (workbench.html)│◄──►│   (nginx.conf)   │◄──►│   (Database)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ PhotoUploadService │   │   Main API       │   │   AuthService   │
│ (photo_upload.py)  │   │  (app.py)        │   │  (auth.py)      │
└─────────────────┘   └──────────────────┘   └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                      ┌──────────────────┐
                      │   Celery Worker  │
                      │  (worker.py)     │
                      └──────────────────┘
                              │
                              ▼
                   ┌───────────────────────┐
                   │  Computer Vision Model │
                   │   (cv_model.py)       │
                   └───────────────────────┘
```

## Установка и запуск

### Требования

- Python 3.8+
- Docker и Docker Compose
- PostgreSQL
- Redis

### Быстрый запуск с Docker Compose

Для быстрого запуска всех сервисов используйте Docker Compose:

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd <project-directory>

# Запустите все сервисы
docker-compose up -d

# Приложение будет доступно по адресу: http://localhost:8080
```

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

# Запуск сервиса аутентификации
uvicorn full-stack.AuthService.auth:app --host 0.0.0.0 --port 8002

# Запуск Celery worker
celery -A src.tasks.worker.celery_app worker --loglevel=info
```

## Использование

### Аутентификация

Для работы с API необходимо авторизоваться. Все endpoints требуют наличия cookie с session_token.

#### Регистрация нового пользователя
```http
POST /api/register
Content-Type: application/json

{
  "name": "username",
  "email": "user@example.com",
  "password": "password"
}
```

#### Вход в систему
```http
POST /api/login
Content-Type: application/json

{
  "name": "username",
  "password": "password"
}
```

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

### AuthService

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Проверка состояния сервиса |
| `/api/auth` | GET/POST | Проверка аутентификации |
| `/api/register` | POST | Регистрация нового пользователя |
| `/api/login` | POST | Вход в систему |

## Структура проекта

```
├── src/                    # Основной исходный код
│   ├── api/               # API endpoints
│   │   ├── app.py         # Основное приложение FastAPI
│   │   └── integration.py # Интеграционные endpoints
│   ├── models/            # Модели машинного обучения
│   │   ├── cv_model.py    # Модель компьютерного зрения
│   │   └── feature_extractor.py # Извлечение признаков
│   ├── geo/               # Геокодирование
│   │   └── geocoder.py    # Геокодер
│   ├── data/              # Работа с данными
│   ├── engine/            # Движок обработки
│   ├── tasks/             # Асинхронные задачи (Celery)
│   │   └── worker.py      # Celery worker
│   ├── utils/             # Вспомогательные функции
│   └── visualization/     # Визуализация
├── full-stack/            # Микросервисы
│   ├── PhotoUploadService/ # Сервис загрузки фотографий
│   │   ├── photo_upload.py # Основной сервис
│   │   └── Dockerfile     # Docker конфигурация
│   ├── AuthService/       # Сервис аутентификации
│   │   ├── auth.py        # Основной сервис
│   │   └── Dockerfile     # Docker конфигурация
│   ├── DB/               # Конфигурация базы данных
│   └── ngnix/            # Веб-интерфейс и reverse proxy
├── notebooks/            # Jupyter notebooks для исследований
├── tests/                # Тесты
├── configs/              # Конфигурационные файлы
├── data/                 # Данные проекта
│   ├── index/            # FAISS индексы
│   └── processed/        # Обработанные данные
└── logs/                 # Логи приложения
```

## База данных

Система использует PostgreSQL для хранения метаданных изображений и результатов обработки.

### Основные таблицы

1. **users** - информация о пользователях
2. **photos** - загруженные фотографии
3. **processing_results** - результаты обработки изображений
4. **query_history** - история запросов пользователей

### Миграции базы данных

Для инициализации базы данных выполните SQL скрипты из директории `src/sql_scripts/`.

## Модели машинного обучения

### Computer Vision Model

Модель на основе GeoCLIP для определения координат зданий на изображениях.

#### Особенности:
- Использует FAISS индекс для поиска похожих изображений
- Интегрирована с внешними геокодерами
- Поддерживает асинхронную обработку

#### Загрузка данных из S3

Модель загружает необходимые данные из S3 по следующим путям:
- `processed_data/models/index/faiss_index.bin` - FAISS индекс
- `processed_data/models/index/image_mapping.csv` - Маппинг изображений
- `processed_data/moscow_images.csv` - Метаданные изображений

## Разработка

### Структура кода

Код следует принципам чистой архитектуры:
- Разделение на слои (presentation, business logic, data access)
- Использование dependency injection
- Четкое разделение ответственности между модулями

### Добавление новых функций

1. Создайте новую ветку для разработки
2. Реализуйте функционал в соответствующем модуле
3. Напишите тесты
4. Обновите документацию
5. Создайте pull request

### Тестирование

Для запуска тестов используйте:
```bash
pytest tests/
```

### Логирование

Система использует стандартное логирование Python. Уровень логирования можно настроить через переменные окружения.

## Производительность

### Масштабирование

Система спроектирована для масштабирования:
- Горизонтальное масштабирование сервисов
- Асинхронная обработка задач через Celery
- Кэширование результатов

### Оптимизация

- Использование FAISS для быстрого поиска похожих изображений
- Оптимизация загрузки изображений из S3
- Кэширование часто используемых данных

## Безопасность

- Аутентификация через session tokens
- Защита от SQL-инъекций
- Валидация входных данных
- HTTPS в production среде

## Мониторинг и логирование

- Централизованное логирование
- Метрики производительности
- Алертинг о критических ошибках

## Лицензия

Проект распространяется под лицензией MIT.
