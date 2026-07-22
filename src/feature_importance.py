import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load tuned model
model = joblib.load(
    BASE_DIR / "models" / "best_model_tuned.pkl"
)

# Load preprocessor
preprocessor = joblib.load(
    BASE_DIR / "models" / "preprocessor.pkl"
)

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Get feature importances
importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort descending
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Save CSV
importance_df.to_csv(
    BASE_DIR / "models" / "feature_importance.csv",
    index=False
)

print(importance_df.head(20))

top_features = importance_df.head(15)

plt.figure(figsize=(10,6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Important Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    BASE_DIR / "reports" / "feature_importance.png",
    dpi=300
)

plt.show()