FROM python:3.9-slim

WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код  API и папки training/config для загрузки модели из MLflow
COPY api/ ./api/
COPY training/ ./training/

# Указываем порт, который будет слушать приложение внутри контейнера
EXPOSE 8000

# Команда для запуска API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]