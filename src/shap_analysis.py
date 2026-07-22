import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load model
model = joblib.load(
    BASE_DIR / "models" / "best_model_tuned.pkl"
)

# Load preprocessor
preprocessor = joblib.load(
    BASE_DIR / "models" / "preprocessor.pkl"
)

# Load dataset
df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "cleaned_ballistic_v50.csv"
)

X = df.drop("v50", axis=1)

# Transform features
X_processed = preprocessor.transform(X)

# Feature names
feature_names = preprocessor.get_feature_names_out()

# SHAP Explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_processed)

plt.figure(figsize=(10,8))

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    BASE_DIR / "reports" / "shap_summary.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(10,8))

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    BASE_DIR / "reports" / "shap_bar.png",
    dpi=300
)

plt.show()

