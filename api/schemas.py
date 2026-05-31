from pydantic import BaseModel

class PredictionRequest(BaseModel):
    Age: int
    MonthlyIncome: float
    OverTime: str  # "Yes" или "No"
    # В реальном приложении добавьте все признаки, которые были в обучении

class PredictionResponse(BaseModel):
    attrition_probability: float
    prediction: int  # 0 или 1