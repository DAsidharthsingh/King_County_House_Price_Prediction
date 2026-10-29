import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import prepare_data


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def prepare_train_test_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X, y, model_name):
    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y, predictions)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Linear Regression"
    )

    return model, metrics


def train_lasso_regression(X_train, X_test, y_train, y_test):
    parameter_grid = {
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=Lasso(max_iter=10000),
        param_grid=parameter_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        "Lasso Regression"
    )

    return best_model, metrics, grid_search.best_params_


def train_ridge_regression(X_train, X_test, y_train, y_test):
    parameter_grid = {
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=Ridge(),
        param_grid=parameter_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        "Ridge Regression"
    )

    return best_model, metrics, grid_search.best_params_


def compare_models(results):
    return (
        pd.DataFrame(results)
        .sort_values(by="R2", ascending=False)
        .reset_index(drop=True)
    )


def save_model(model, scaler, feature_names):
    """Save the trained model, scaler, and feature names."""

    joblib.dump(
        model,
        MODEL_DIR / "lasso_model.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    joblib.dump(
        list(feature_names),
        MODEL_DIR / "feature_names.pkl"
    )

    print("\nModel artifacts saved successfully.")
    print(f"Model: {MODEL_DIR / 'lasso_model.pkl'}")
    print(f"Scaler: {MODEL_DIR / 'scaler.pkl'}")
    print(f"Features: {MODEL_DIR / 'feature_names.pkl'}")


if __name__ == "__main__":

    X, y = prepare_data("Data/house_sales.csv")

    X_train, X_test, y_train, y_test = prepare_train_test_data(
        X,
        y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    results = []

    linear_model, linear_metrics = train_linear_regression(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    results.append(linear_metrics)

    lasso_model, lasso_metrics, lasso_params = train_lasso_regression(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    results.append(lasso_metrics)

    ridge_model, ridge_metrics, ridge_params = train_ridge_regression(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    results.append(ridge_metrics)

    comparison = compare_models(results)

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    print("\nBest Lasso Parameters:")
    print(lasso_params)

    print("\nBest Ridge Parameters:")
    print(ridge_params)

    best_model_name = comparison.iloc[0]["Model"]

    if best_model_name == "Lasso Regression":
        save_model(lasso_model, scaler, X.columns)

    elif best_model_name == "Ridge Regression":
        joblib.dump(
            ridge_model,
            MODEL_DIR / "ridge_model.pkl"
        )
        joblib.dump(
            scaler,
            MODEL_DIR / "scaler.pkl"
        )
        print("\nRidge model and scaler saved successfully.")

    else:
        joblib.dump(
            linear_model,
            MODEL_DIR / "linear_model.pkl"
        )
        joblib.dump(
            scaler,
            MODEL_DIR / "scaler.pkl"
        )
        print("\nLinear Regression model and scaler saved successfully.")