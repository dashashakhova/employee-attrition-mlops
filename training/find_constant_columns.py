import pandas as pd

df = pd.read_csv("/Users/dariashakhova/Desktop/MIPT/Развертывание_ML/employee-attrition-mlops/data/raw/hr_attrition.csv")

for col in df.columns:
    unique_count = df[col].nunique()

    if unique_count == 1:
        print(col)