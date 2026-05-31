import pandas as pd

df = pd.read_csv(
    "/Users/dariashakhova/Desktop/MIPT/Развертывание_ML/employee-attrition-mlops/data/raw/hr_attrition.csv"
)

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

for col in [
    "OverTime",
    "JobSatisfaction",
    "WorkLifeBalance"
]:
    print(f"\n===== {col} =====")

    result = (
        pd.crosstab(
            df[col],
            df["Attrition"],
            normalize="index"
        ) * 100
    )

    print(result)