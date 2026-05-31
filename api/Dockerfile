FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt.txt
COPY api/ ./api/
COPY training/ ./training/
COPY data/ ./data/
COPY models/ ./models/
ENV MLFLOW_TRACKING_URI=http://mlflow:8080
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]