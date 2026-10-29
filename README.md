# 🏠 King County House Price Prediction

A machine learning regression project for predicting house sale prices using property characteristics from the King County housing dataset.

The project implements data preprocessing, exploratory data analysis, feature engineering, feature scaling, regression model training, hyperparameter tuning, model comparison, and price prediction using a trained Lasso Regression model.

---

## 📌 Project Overview

House prices depend on multiple factors such as property size, number of bedrooms and bathrooms, building grade, construction year, location-related information, property value, and other characteristics.

The objective of this project is to build and evaluate regression models that can predict the sale price of a house based on these features.

### Problem Type

**Supervised Machine Learning — Regression**

### Target Variable

`SalePrice`

### Models Implemented

- Linear Regression
- Lasso Regression
- Ridge Regression

### Best Performing Model

**Lasso Regression**

---

## 🎯 Objectives

- Clean and preprocess the housing dataset.
- Perform feature engineering on date-related variables.
- Handle duplicate records and target outliers.
- Explore relationships between housing features and sale prices.
- Train multiple regression models.
- Tune Lasso and Ridge hyperparameters using GridSearchCV.
- Compare models using regression evaluation metrics.
- Save the best-performing model for future predictions.
- Build a reusable prediction module.

---

## 📂 Project Structure

```text
King_County_House_Sale_Prediction/
│
├── Data/
│   └── house_sales.csv
│
├── Notebooks/
│   └── King_County_House_Sale_prediction.ipynb
│
├── models/
│   ├── lasso_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── model_training.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Jupyter Notebook** for the reference notebook

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Cleaning
     │
     ├── Duplicate Removal
     ├── Date Feature Extraction
     ├── Categorical Encoding
     └── Boolean Conversion
     │
     ▼
Outlier Removal
     │
     ▼
Feature / Target Separation
     │
     ▼
Train-Test Split
     │
     ▼
Feature Scaling
     │
     ▼
Model Training
     │
     ├── Linear Regression
     ├── Lasso Regression
     └── Ridge Regression
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Model Serialization
     │
     ▼
House Price Prediction
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

### 1. Duplicate Removal

Duplicate records are removed from the dataset before model training.

### 2. Date Feature Engineering

The following transformations are performed:

- `DocumentDate` → `DocumentYear`, `DocumentMonth`
- `ym` → `SaleYear`, `SaleMonth`

The original date columns are then removed.

### 3. Property Type Encoding

`PropertyType` is encoded numerically:

| Property Type | Encoded Value |
|---|---:|
| Single Family | 0 |
| Multiplex | 1 |
| Townhouse | 2 |

### 4. Boolean Conversion

Boolean columns are converted into numerical `0/1` representations.

### 5. Outlier Removal

The Interquartile Range (IQR) method is applied to `SalePrice`.

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Records outside these limits are removed.

---

## 📊 Exploratory Data Analysis

The project includes visualizations for understanding the dataset and relationships between variables.

### Visualizations

- Sale Price Distribution
- Living Area vs Sale Price
- Correlation Heatmap
- Yearly Average Sale Price
- Sale Price by Building Grade
- Bedroom Distribution
- Building Grade Distribution

These visualizations help identify patterns, correlations, distributions, and relationships between property characteristics and sale prices.

---

## 🤖 Machine Learning Models

### 1. Linear Regression

Used as the baseline regression model.

### 2. Lasso Regression

Lasso Regression applies L1 regularization and was tuned using GridSearchCV.

Hyperparameter search:

```python
alpha = [0.001, 0.01, 0.1, 1, 10, 100]
```

Configuration:

```text
Cross-validation: 5-fold
Scoring: R²
Parallel processing: n_jobs=-1
```

### 3. Ridge Regression

Ridge Regression applies L2 regularization and was tuned using the same alpha search space.

---

## 📏 Model Evaluation

The models are evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted prices.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

### Root Mean Squared Error (RMSE)

The square root of MSE, providing the error in the same unit as the target variable.

### R² Score

Measures how much of the variance in the target variable is explained by the model.

---

## 📈 Model Performance

The models were evaluated using the cleaned and preprocessed dataset.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| **Lasso Regression** | **11,994.75** | **3.095 × 10⁸** | **17,592.82** | **0.989708** |
| Ridge Regression | 11,998.16 | 3.096 × 10⁸ | 17,596.48 | 0.989704 |
| Linear Regression | 11,998.16 | 3.096 × 10⁸ | 17,596.48 | 0.989704 |

### Best Model

**Lasso Regression**

Best hyperparameter:

```text
alpha = 100
```

The Lasso model achieved an R² score of approximately **0.9897** on the test set.

---

## 💾 Saved Model Artifacts

The trained model and preprocessing artifacts are stored in the `models/` directory.

```text
models/
├── lasso_model.pkl
├── scaler.pkl
└── feature_names.pkl
```

### `lasso_model.pkl`

Contains the trained Lasso Regression model.

### `scaler.pkl`

Contains the fitted `StandardScaler` used during model training.

### `feature_names.pkl`

Stores the exact feature order used by the trained model.

This ensures that prediction inputs are passed to the model in the same order as during training.

---

## 🔮 House Price Prediction

The prediction module loads the saved model and scaler and performs prediction on input features.

Run:

```bash
python src/predict.py
```

Example prediction result from a test house:

```text
Predicted House Price: $286,112.74
Actual Sale Price:     $280,000.00
Absolute Error:          $6,112.74
```

The test prediction had an absolute error of approximately **$6,112.74**.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd King_County_House_Sale_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run preprocessing

```bash
python src/data_preprocessing.py
```

### 4. Train the models

```bash
python src/model_training.py
```

This trains the regression models, performs hyperparameter tuning, compares their performance, and saves the best model artifacts.

### 5. Run prediction

```bash
python src/predict.py
```

---

## 📓 Notebook

The original notebook is retained in:

```text
Notebooks/King_County_House_Sale_prediction.ipynb
```

It provides the notebook-based documentation and exploratory analysis associated with the project, while the `src/` directory contains the cleaned modular Python implementation.

---

## 🧠 Key Learnings

This project demonstrates:

- Regression-based machine learning
- Data cleaning and preprocessing
- Feature engineering
- Outlier detection using IQR
- Exploratory Data Analysis
- Feature scaling
- Regularization techniques
- L1 and L2 regularization
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Regression model evaluation
- Model serialization with Joblib
- Building a reusable prediction pipeline

---

## 🚀 Future Improvements

Possible future improvements include:

- Adding additional regression algorithms such as Random Forest and Gradient Boosting.
- Building a web-based prediction interface using Streamlit.
- Adding automated model training and evaluation.
- Creating more advanced feature engineering pipelines.
- Adding model explainability using SHAP or similar techniques.
- Deploying the prediction application as a web service.

---

## 👨‍💻 Author

**Sidharth Singh**

B.Tech — Computer Science Engineering (Data Science)

---

## 📄 License

This project is intended for educational and academic purposes.