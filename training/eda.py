import pandas as pd

df = pd.read_csv("/Users/dariashakhova/Desktop/MIPT/Развертывание_ML/employee-attrition-mlops/data/raw/hr_attrition.csv")

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== INFO ===")
print(df.info())

print("\n=== TARGET ===")
print(df["Attrition"].value_counts())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())