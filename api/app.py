from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
from api.model_loader import get_model
import pandas as pd
import logging
from training.config import RAW_DATA_PATH

app = FastAPI(title="Employee Attrition Prediction API")
logger = logging.getLogger(__name__)

# Глобальные переменные для значений по умолчанию
default_values = None
all_features = None

@app.on_event("startup")
async def startup_event():
    global default_values, all_features
    logger.info("Loading model from MLflow Registry...")
    get_model()
    logger.info("Loading training data to compute default values...")
    df = pd.read_csv(RAW_DATA_PATH)
    # Удаляем те же колонки, что и в baseline_model (кроме целевой)
    drop_columns = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber", "Attrition"]
    df = df.drop(columns=drop_columns, errors='ignore')
    all_features = df.columns.tolist()
    default_values = {}
    for col in all_features:
        if df[col].dtype in ['int64', 'float64']:
            default_values[col] = df[col].median()
        else:
            default_values[col] = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
    logger.info(f"Loaded {len(all_features)} features with default values")
    logger.info("Startup complete")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": get_model() is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        model = get_model()
        input_dict = request.dict()
        full_input = {}
        for col in all_features:
            if col in input_dict:
                full_input[col] = input_dict[col]
            else:
                full_input[col] = default_values[col]
        df_input = pd.DataFrame([full_input])
        proba = model.predict_proba(df_input)[0][1]
        pred = int(proba >= 0.5)
        return PredictionResponse(attrition_probability=round(proba, 4), prediction=pred)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))