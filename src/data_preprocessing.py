import pandas as pd 

def load_dataset(file_path :str)-> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(df:pd.DataFrame)-> pd.DataFrame:
    df=df.copy()
    
    df=df.drop_duplicates()

    if "DocumentDate" in df.columns:
        df["DocumentDate"] = pd.to_datetime(df["DocumentDate"],errors="coerce")
        df["DocumentYear"] = df["DocumentDate"].dt.year
        df["DocumentMonth"] = df["DocumentDate"].dt.month
        df = df.drop(columns=["DocumentDate"])

    if "ym" in df.columns:
        df["ym"] = pd.to_datetime(df["ym"], errors="coerce")
        df["SaleYear"] = df["ym"].dt.year
        df["SaleMonth"] = df["ym"].dt.month
        df = df.drop(columns=["ym"])

    df["PropertyType"] = (
    df["PropertyType"]
    .map({
        "Multiplex": 1,
        "Single Family": 0,
        "Townhouse": 2
    })
    .astype("Int64"))

    boolean_columns = df.select_dtypes(include=["bool"]).columns

    for column in boolean_columns:
        df[column] = df[column].astype(int)

    return df

def remove_outliers(
    df: pd.DataFrame,
    column: str = "SalePrice") -> pd.DataFrame:

    df = df.copy()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ]

    return df

def prepare_data(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Complete preprocessing pipeline and return features and target."""
    df = load_dataset(file_path)
    df = clean_data(df)
    df = remove_outliers(df)

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    return X, y

if __name__ == "__main__":
    X, y = prepare_data("Data/house_sales.csv")

    print("Preprocessing completed successfully.")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
