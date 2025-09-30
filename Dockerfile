# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p data/processed data/index logs

# Открываем порт для API
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
