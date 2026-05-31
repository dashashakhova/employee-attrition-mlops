import pandas as pd


DROP_COLUMNS = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber"
]


def load_data(path):
    df = pd.read_csv(path)

    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    df = df.drop(columns=DROP_COLUMNS)

    return df