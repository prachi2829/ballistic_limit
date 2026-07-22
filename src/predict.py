import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "best_model_tuned.pkl"
)

preprocessor = joblib.load(
    BASE_DIR / "models" / "preprocessor.pkl"
)

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "cleaned_ballistic_v50.csv"
)

material_lookup = (
    df.groupby("Material")
    .first()[
        [
            "density",
            "modulus",
            "hardness",
            "yield",
            "PR",
            "UTS",
            "elongation",
            "Ludwik-n"
        ]
    ]
)

projectile_lookup = (
    df.groupby("projectile")
    .first()[
        [
            "proj_type",
            "calibre",
            "proj_mass",
            "proj_density",
            "proj_hardness"
        ]
    ]
)

def predict_v50(
    projectile,
    material,
    thickness,
    angle
):
    
    proj = projectile_lookup.loc[projectile]
    mat = material_lookup.loc[material]

    input_df = pd.DataFrame({
    "projectile": [projectile],
    "proj_type": [proj["proj_type"]],
    "calibre": [proj["calibre"]],
    "proj_mass": [proj["proj_mass"]],
    "proj_density": [proj["proj_density"]],
    "proj_hardness": [proj["proj_hardness"]],
    "Material": [material],
    "thickness": [thickness],
    "density": [mat["density"]],
    "modulus": [mat["modulus"]],
    "hardness": [mat["hardness"]],
    "yield": [mat["yield"]],
    "PR": [mat["PR"]],
    "UTS": [mat["UTS"]],
    "elongation": [mat["elongation"]],
    "Ludwik-n": [mat["Ludwik-n"]],
    "angle": [angle]
    })

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)[0]

    return round(float(prediction), 2)



if __name__ == "__main__":
    v50 = predict_v50(
        projectile="20MMFSP",
        material="AA7085-T721",
        thickness=25,
        angle=0
    )

    print(v50)