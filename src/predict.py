import joblib
import pandas as pd
from pathlib import Path

from data_preprocessing import clean_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def load_artifacts():
    """Load the trained model, scaler, and feature names."""

    model = joblib.load(MODEL_DIR / "lasso_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    return model, scaler, feature_names


def predict_house_price(input_data: dict) -> float:
    """
    Predict house sale price from user-provided features.
    """

    model, scaler, feature_names = load_artifacts()

    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame([input_data])

    # Check for missing features
    missing_features = [
        feature
        for feature in feature_names
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Keep exactly the same feature order used during training
    input_df = input_df[feature_names]

    # Scale features using the trained scaler
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    return float(prediction[0])


def get_sample_input():
    """
    Return a sample house input for testing.
    """

    return {
        "PropertyID": 1000102.0,
        "PropertyType": 1.0,
        "zhvi_px": 405100.0,
        "zhvi_idx": 0.930836,
        "AdjSalePrice": 300805.0,
        "NbrLivingUnits": 2.0,
        "SqFtLot": 9373.0,
        "SqFtTotLiving": 2400.0,
        "SqFtFinBasement": 0.0,
        "Bathrooms": 3.0,
        "Bedrooms": 6.0,
        "BldgGrade": 7.0,
        "YrBuilt": 1991.0,
        "YrRenovated": 0.0,
        "TrafficNoise": 0.0,
        "LandVal": 70000.0,
        "ImpsVal": 229000.0,
        "ZipCode": 98002.0,
        "NewConstruction": 0.0,
        "DocumentYear": 2014.0,
        "DocumentMonth": 9.0,
        "SaleYear": 2014.0,
        "SaleMonth": 9.0
    }


if __name__ == "__main__":

    print("King County House Price Prediction")
    print("=" * 40)

    sample_house = get_sample_input()

    try:
        predicted_price = predict_house_price(sample_house)

        print("\nSample House Prediction")
        print("-" * 40)
        print(f"Predicted Sale Price: ${predicted_price:,.2f}")

    except Exception as error:
        print(f"\nPrediction failed: {error}")