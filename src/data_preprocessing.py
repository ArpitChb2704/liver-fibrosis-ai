import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np

def add_fib4(df):
    df["FIB4"] = (df["Age"] * df["AST"]) / (df["PLT"] * np.sqrt(df["ALT"] + 1e-6))
    return df

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(".", "_", regex=False)
    return df

def preprocess_data(df):
    # Drop unwanted index column if exists
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # add FIB-4 score as a feature
    df = add_fib4(df)

    # Target column
    target_col = "group"

    # Convert labels to binary
    df[target_col] = df[target_col].apply(lambda x: 0 if x == "S0_2" else 1)

    # Split
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Remove non-numeric columns if any
    X = X.select_dtypes(include=["number"])

    # Handle missing values
    X = X.fillna(X.mean())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame to keep column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler