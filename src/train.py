import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_ballistic_v50.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

X = df.drop("v50", axis=1)
y = df["v50"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

preprocessor = joblib.load(PREPROCESSOR_PATH)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        random_state=42
    )
}


results = []


best_model = None
best_model_name = None
best_r2 = -np.inf

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_processed, y_train)
    trained_models[name] = model

    predictions = model.predict(X_test_processed)
    print("Completed")

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name




results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========\n")
print(results_df)

# Save best model
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
joblib.dump(best_model, MODEL_PATH)

# Save model comparison table
RESULTS_PATH = BASE_DIR / "models" / "model_results.csv"
results_df.to_csv(RESULTS_PATH, index=False)

# Get metrics of best model
best_metrics = results_df[results_df["Model"] == best_model_name].iloc[0]

# Save best model information
BEST_MODEL_INFO = BASE_DIR / "models" / "best_model_info.txt"

with open(BEST_MODEL_INFO, "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"R2 Score: {best_metrics['R2 Score']}\n")
    f.write(f"MAE: {best_metrics['MAE']}\n")
    f.write(f"RMSE: {best_metrics['RMSE']}\n")

print("\n======================================")
print(f"Best Model : {best_model_name}")
print(f"R² Score   : {best_metrics['R2 Score']}")
print("======================================")

print("\nSaved:")
print("best_model.pkl")
print("model_results.csv")
print("best_model_info.txt")