# Хакатон по цифровой трансформации от правительства Москвы

Разработка полноценного MVP продукта для определения местоположения по фотографиям с собственным сайтом для работы с пользователем

## 📁 Структура проекта

```
hack_digital_transformation/
├── configs/                    # Конфигурационные файлы
│   └── config.yaml            # Основной конфигурационный файл
├── data/                      # Данные (управляются через DVC)
│   ├── raw/                   # Сырые данные
│   └── processed/             # Обработанные данные
├── logs/                      # Логи выполнения
├── models/                    # Обученные модели (DVC)
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
- DVC (для управления данными и моделями)
- Make (рекомендуется)

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

# Очистка временных файлов
make clean
```

### Работа с данными

Данные и модели управляются через DVC:

```bash
# Добавление файла данных под версионный контроль
dvc add data/raw/dataset.csv

# Синхронизация с удаленным хранилищем
dvc push

# Получение последней версии данных
dvc pull

# Проверка статуса данных
dvc status
```

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

Проект настроен для работы с MLflow для отслеживания экспериментов:

```python
import mlflow

# Автоматическое логирование параметров и метрик
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Ваш код обучения
    model.fit(X_train, y_train)
```

Для просмотра результатов:

```bash
mlflow ui
```

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
