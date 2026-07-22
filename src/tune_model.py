import joblib
import optuna
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "cleaned_ballistic_v50.csv"
)

X = df.drop("v50", axis=1)
y = df["v50"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor = joblib.load(
    BASE_DIR / "models" / "preprocessor.pkl"
)

X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0,
            5
        ),

        "objective": "reg:squarederror",

        "random_state": 42,

        "verbosity": 0
    }

    model = XGBRegressor(**params,n_jobs=-1)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    return scores.mean()

print("\nStarting Hyperparameter Optimization...\n")

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

study.trials_dataframe().to_csv(
    BASE_DIR / "models" / "optuna_trials.csv",
    index=False
)

best_params = study.best_params
best_cv_score = study.best_value

best_params["objective"] = "reg:squarederror"
best_params["random_state"] = 42
best_params["eval_metric"] = "rmse"

best_model = XGBRegressor(
    **best_params,
    n_jobs=-1
)

best_model.fit(
    X_train,
    y_train
)

predictions = best_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== OPTUNA RESULTS ==========\n")

print("Best Parameters:\n")

for key, value in best_params.items():
    print(f"{key}: {value}")

print("\nPerformance on Test Set")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print("\nBaseline XGBoost R² : 0.8902")
print(f"Tuned XGBoost R²    : {r2:.4f}")
print(f"Improvement         : {r2 - 0.8902:.4f}")
print(f"\nBest Cross Validation R² : {best_cv_score:.4f}")

PARAMS_PATH = BASE_DIR / "models" / "best_xgboost_params.txt"

with open(PARAMS_PATH, "w") as f:

    f.write("Best XGBoost Parameters\n\n")

    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")

    f.write("\n")

    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"Cross Validation R2: {best_cv_score:.4f}\n")

joblib.dump(
    best_model,
    BASE_DIR / "models" / "best_model_tuned.pkl"
)

print("\nTuned model saved successfully!")
