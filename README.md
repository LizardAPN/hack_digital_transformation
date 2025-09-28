# Building Detector API

Система автоматического определения географических координат и адресов зданий на фотоматериалах с использованием визуального поиска и компьютерного зрения.

## 🏗️ Архитектура решения

Основная идея: Визуальный поиск по базе изображений (Visual Image Retrieval)

Вместо прямого сравнения пикселей, мы сравниваем визуальные признаки (features) изображений.

### Алгоритм работы:

#### Фаза 1: Подготовка базы данных
1. Извлечение признаков из всех фото Москвы (предобученная CNN - ResNet50)
2. Создание поискового индекса (FAISS или Annoy)
3. Создание маппинга: вектор → имя файла в S3 → координаты

#### Фаза 2: Обработка нового изображения
1. Извлечение признаков из загруженной фотографии
2. Поиск K ближайших соседей в FAISS индексе
3. Ранжирование результатов по косинусной близости
4. Определение координат на основе топ-N самых похожих изображений

## 🚀 Быстрый старт

### Запуск с помощью Docker

```bash
# Клонирование репозитория
git clone <repository-url>
cd building-detector

# Создание .env файла (см. .env.example)
cp .env.example .env
# Отредактируйте .env файл с вашими настройками

# Запуск всех сервисов
docker-compose up -d

# Сервисы будут доступны:
# API: http://localhost:8000
# PostgreSQL: localhost:5432
# Redis: localhost:6379
```

### Локальный запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск API сервера
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Запуск Celery worker
celery -A src.tasks.worker worker --loglevel=info

# Запуск Celery beat (для планирования задач)
celery -A src.tasks.worker beat --loglevel=info
```

## 📡 API Endpoints

### Синхронная обработка

- `POST /process_image` - Обработка одного изображения
- `POST /predict` - Прогнозирование с использованием обученной модели
- `GET /health` - Проверка состояния сервиса

### Асинхронная обработка

- `POST /process_image_async` - Асинхронная обработка одного изображения
- `POST /process_images_batch` - Пакетная асинхронная обработка изображений
- `GET /task/{task_id}` - Получение статуса задачи
- `GET /tasks/request/{request_id}` - Получение всех задач по request_id
- `GET /results/latest` - Получение последних результатов обработки

### Примеры использования

#### Синхронная обработка изображения:

```bash
curl -X POST "http://localhost:8000/process_image" \
     -H "Content-Type: application/json" \
     -d '{"image_path": "/path/to/image.jpg"}'
```

#### Асинхронная обработка изображения:

```bash
curl -X POST "http://localhost:8000/process_image_async" \
     -H "Content-Type: application/json" \
     -d '{"image_path": "/path/to/image.jpg", "request_id": "unique-request-id"}'
```

#### Пакетная обработка изображений:

```bash
curl -X POST "http://localhost:8000/process_images_batch" \
     -H "Content-Type: application/json" \
     -d '{
       "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
       "request_id": "batch-request-id"
     }'
```

#### Получение статуса задачи:

```bash
curl -X GET "http://localhost:8000/task/{task_id}"
```

## ⚙️ Конфигурация

### Переменные окружения (.env)

```env
# AWS S3 Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_BUCKET_NAME=your_bucket_name

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=building_detector

# Geocoders
YANDEX_GEOCODER_API_KEY=your_yandex_api_key
GOOGLE_GEOCODER_API_KEY=your_google_api_key

# Celery
CELERY_WORKER_CONCURRENCY=4
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/building_detector
```

## 📊 Производительность

Система оптимизирована для обработки 1000 изображений за ≤3 часов:

- Параллельная обработка с помощью Celery workers
- Кэширование признаков изображений
- Оптимизированный FAISS индекс для быстрого поиска
- Асинхронная обработка с отслеживанием прогресса

## 🛠️ Структура проекта

```
building-detector/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── models/           # Computer vision models
│   ├── geo/              # Geocoding services
│   ├── data/             # Data processing utilities
│   ├── tasks/            # Celery tasks
│   ├── utils/            # Utility functions
│   └── sql_scripts/      # Database initialization scripts
├── data/                 # Local data storage
├── logs/                 # Application logs
├── tests/                # Unit and integration tests
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Services orchestration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🧪 Тестирование

```bash
# Запуск unit тестов
python -m pytest tests/

# Запуск тестов с покрытием
python -m pytest tests/ --cov=src
```

## 🔄 CI/CD и Линтеры

Проект использует GitHub Actions для непрерывной интеграции и доставки. Конфигурация находится в `.github/workflows/ci.yml`.

### Линтеры и форматирование кода

Проект использует несколько линтеров для обеспечения качества кода:

- **Black** - автоформатирование кода (максимальная длина строки: 120)
- **Isort** - сортировка импортов
- **Ruff** - быстрая проверка кода
- **Flake8** - дополнительные проверки стиля

### Локальное использование линтеров

```bash
# Форматирование кода с помощью black и isort
black .
isort .

# Проверка кода с помощью ruff
ruff check .

# Проверка стиля кода с помощью flake8
flake8 .
```

### Автоматическая проверка

При каждом пуше или создании pull request запускается pipeline, который:
1. Проверяет форматирование и стиль кода
2. Запускает unit тесты
3. Собирает пакет для распространения

## 📈 Мониторинг

Система включает в себя:

- Логирование всех операций
- Отслеживание прогресса асинхронных задач
- Статистика обработки изображений
- Мониторинг ошибок и исключений

## 🤝 Интеграции

- **S3 Storage**: Хранение изображений и метаданных
- **FAISS**: Быстрый поиск по визуальным признакам
- **Geocoders**: Yandex, Google, OpenStreetMap
- **PostgreSQL**: Хранение результатов обработки
- **Redis**: Очереди задач и кэширование

## 📄 Лицензия

MIT License
