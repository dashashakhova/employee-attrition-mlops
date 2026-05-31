import pandas as pd

df = pd.read_csv(
    "/Users/dariashakhova/Desktop/MIPT/Развертывание_ML/employee-attrition-mlops/data/raw/hr_attrition.csv"
)

for col in [
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "OverTime",
    "JobSatisfaction",
    "WorkLifeBalance"
]:
    print(f"\n=== {col} ===")
    print(df[col].value_counts().head(20))