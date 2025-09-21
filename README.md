# Хакатон по цифровой трансформации от правительства Москвы

Разработка полноценного MVP продукта для определения местоположения по фотографиям с собственным сайтом для работы с пользователем

## 📁 Структура проекта

```
hack_digital_transformation/
├── configs/                    # Конфигурационные файлы
│   └── config.yaml            # Основной конфигурационный файл
├── data/                      # Данные (хранятся локально или в S3)
│   ├── raw/                   # Сырые данные
│   └── processed/             # Обработанные данные
├── logs/                      # Логи выполнения
├── models/                    # Обученные модели 
├── notebooks/                 # Jupyter ноутбуки для исследований
├── plots/                     # Графики и диаграммы
├── reports/                   # Отчеты и визуализации
│   └── figures/               # Фигуры и графики
├── scripts/                   # Скрипты для автоматизации
│   └── setup_environment.sh   # Скрипт настройки окружения
├── src/                       # Исходный код проекта
│   ├── api/                   # API для развертывания модели
│   ├── data/                  # Код для обработки данных
│   ├── features/              # Код для инжиниринга признаков
│   ├── models/                # Код для обучения и оценки моделей
│   ├── sql_scripts/           # SQL скрипты для работы с БД
│   └── visualization/         # Код для визуализации
└── tests/                     # Тесты
```

## ⚡ Быстрый старт

### Предварительные требования

- Python 3.11
- Git
- Make (рекомендуется)
- Доступ к S3 (для работы с данными в облаке)
- MLflow (для отслеживания экспериментов)

### Настройка окружения

#### Клонируйте репозиторий:

```bash
git clone <your-repo-url>
cd hack_digital_transformation
```

#### Запустите скрипт настройки окружения:

```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

#### Активируйте виртуальное окружение:

```bash
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate     # Windows
```

## 🛠 Использование

### Основные команды

```bash
# Установка зависимостей
make install

# Установка зависимостей для разработки
make install-dev

# Установка зависимостей для продакшена
make install-prod

# Запуск тестов
make test

# Проверка кодстайла
make lint

# Форматирование кода
make format

# Подготовка данных
make data

# Обучение модели
make train

# Оценка модели
make evaluate

# Запуск API сервера
make serve

# Сборка Docker образа
make docker

# Запуск интерфейса MLflow
make mlflow-ui

# Очистка временные файлы
make clean
```

### Работа с данными

Данные могут храниться локально или в облачном хранилище S3. Для работы с S3 необходимо настроить учетные данные AWS.

Настройка переменных окружения для работы с S3:
```bash
export S3_BUCKET=your-bucket-name
export S3_RAW_DATA_KEY=data/raw/data.csv
export S3_PROCESSED_PREFIX=data/processed/
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

При запуске скриптов, если указана переменная S3_BUCKET, данные будут автоматически загружаться из S3 и сохраняться в него.
Если переменная S3_BUCKET не указана, данные будут работать локально в директориях `data/raw/` и `data/processed/`.

### Запуск экспериментов

1. Настройте параметры в `configs/config.yaml`
2. Подготовьте данные:
   ```bash
   make data
   ```
3. Обучите модель:
   ```bash
   make train
   ```
4. Оцените модель:
   ```bash
   make evaluate
   ```

Все этапы автоматически логируются в MLflow, если указан URI трекинга:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=my-experiment
```   

### Тестирование

Проект включает автоматические тесты для обеспечения качества кода:

```bash
# Запуск всех тестов
make test

# Запуск тестов с покрытием
pytest tests/ --cov=src

# Запуск конкретного теста
pytest tests/test_data.py::test_load_data
```

## ⚙️ Конфигурация

### Проект использует конфигурационные файлы в формате YAML:

- `configs/config.yaml` - основные настройки проекта

Файл конфигурации содержит следующие секции:

```yaml
# Конфигурация данных
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  train_file: "train.csv"
  test_file: "test.csv"
  target_column: "target"
  test_size: 0.2
  random_state: 42

# Конфигурация фичей
features:
  numeric_features: 
    - "age"
    - "income"
  categorical_features:
    - "gender"
    - "education"

# Конфигурация модели
model:
  name: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

# Конфигурация обучения
training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 5

# Конфигурация логирования
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/training.log"
```

## 📊 Визуализация

Проект включает встроенные возможности для визуализации:

```bash
# Генерация визуализаций
python src/visualization/visualize.py
```

Визуализации сохраняются в:
- `reports/figures/distribution_plots.png` - распределения признаков
- `reports/figures/correlation_heatmap.png` - корреляционная матрица
- `plots/feature_importance.png` - важность признаков
- `reports/figures/roc_curve.png` - ROC кривая

## 📦 Развертывание

### Docker

Соберите и запустите контейнер:

```bash
make docker
docker run -p 8000:8000 hack_digital_transformation
```

### API сервер

Проект включает FastAPI сервер для предсказаний:

```bash
make serve
```

API будет доступно по адресу: http://localhost:8000/docs

Пример запроса к API:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"age": 35, "income": 50000, "gender_M": 1, "education_Master": 0}}'
```

### Cloud развертывание

Пример для AWS SageMaker:

```bash
# Упаковка модели для SageMaker
python src/models/pack_for_sagemaker.py

# Деплой (пример для AWS CLI)
aws s3 sync sagemaker_model/ s3://my-bucket/models/my-model/
```

## 🔬 Эксперименты и отслеживание

Проект использует MLflow для отслеживания экспериментов машинного обучения. При обучении и оценке моделей автоматически логируются:
- Параметры модели и обучения
- Метрики качества
- Артефакты (графики, файлы метрик и т.д.)
- Версии кода

### Настройка MLflow

Для работы с MLflow необходимо указать URI трекинга и имя эксперимента через переменные окружения:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000  # или другой URI вашего MLflow сервера
export MLFLOW_EXPERIMENT_NAME=my-experiment       # имя эксперимента
```

### Запуск MLflow UI

Для просмотра результатов экспериментов можно использовать встроенный интерфейс MLflow:
```bash
make mlflow-ui
```
или напрямую:
```bash
mlflow ui
```

Интерфейс будет доступен по адресу: http://localhost:5000

### Программное использование MLflow

В скриптах обучения и оценки модели уже интегрирована работа с MLflow:
```python
import mlflow

# Автоматическое логирование параметров и метрик scikit-learn моделей
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Ваш код обучения
    model.fit(X_train, y_train)
    # Метрики и параметры автоматически логируются
```

### Хранение артефактов

Артефакты (модели, графики, метрики) автоматически сохраняются как локально, так и в MLflow, если указан URI трекинга.

## 📝 Скрипты

В папке `scripts/` размещаются дополнительные скрипты для автоматизации распространенных сценариев:

- `setup_environment.sh` - настройка окружения с установкой UV и зависимостей
- Дополнительные скрипты можно добавлять по необходимости

## 🧪 CI/CD

Проект включает настройки для непрерывной интеграции и доставки:

- GitHub Actions workflows для тестирования и развертывания
- Pre-commit hooks для проверки кода перед коммитом
- Dockerfile для контейнеризации приложения

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для функциональности (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Добавить замечательную функцию'`)
4. Запушьте ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для подробностей.
