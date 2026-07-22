import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Load cleaned dataset
DATA_PATH = "data/processed/cleaned_ballistic_v50.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())

# Target variable
y = df["v50"]

# Feature variables
X = df.drop(columns=["v50"])

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

preprocessor.fit(X_train)

os.makedirs("models", exist_ok=True)

joblib.dump(preprocessor, "models/preprocessor.pkl")

print("Preprocessor saved successfully!")
