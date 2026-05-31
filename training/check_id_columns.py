import pandas as pd

df = pd.read_csv(
    "/Users/dariashakhova/Desktop/MIPT/Развертывание_ML/employee-attrition-mlops/data/raw/hr_attrition.csv"
)

for col in df.columns:
    if "id" in col.lower() or "number" in col.lower():
        print(col)