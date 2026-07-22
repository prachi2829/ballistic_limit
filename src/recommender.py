import pandas as pd

from src.predict import predict_v50, df

all_materials = sorted(df["Material"].unique())

ALUMINIUM = [
    material
    for material in all_materials
    if material.startswith("AA")
]

STEEL = [
    material
    for material in all_materials
    if not material.startswith("AA")
]

def recommend_design(
    desired_v50,
    projectile,
    material_family="Any",
    max_thickness=None,
    angle=0
):
    
    if material_family == "Aluminium":
        materials = ALUMINIUM

    elif material_family == "Steel":
        materials = STEEL

    else:
        materials = all_materials
    
    thickness_values = sorted(df["thickness"].unique())
    if max_thickness is not None:
        thickness_values = [
        t for t in thickness_values
        if t <= max_thickness
        ]

    recommendations = []

    for material in materials:

        for thickness in thickness_values:

            predicted = predict_v50(
                projectile,
                material,
                thickness,
                angle
            )

            recommendations.append({
            "Material": material,
            "Family": "Aluminium" if material.startswith("AA") else "Steel",
            "Thickness (mm)": thickness,
            "Target V50": desired_v50,
            "Predicted V50": predicted,
            "Difference": round(abs(predicted - desired_v50), 2)
            })
            
    recommendations = pd.DataFrame(recommendations)

    # Sort by smallest prediction error
    recommendations = recommendations.sort_values(
        "Difference"
    )

    # Keep only the best recommendation for each material
    recommendations = recommendations.drop_duplicates(
        subset="Material",
        keep="first"
    )

    # Reset index
    recommendations = recommendations.reset_index(drop=True)

    best = recommendations.iloc[0]

    if best["Difference"] > 50:
        print(
            "Warning: No configuration closely matches the requested V50. "
            "Consider increasing the maximum thickness or selecting a different material family."
        )

    return recommendations.head(3)

    
if __name__ == "__main__":

    print("Aluminium Recommendations")
    result = recommend_design(
        desired_v50=800,
        projectile="20MMFSP",
        material_family="Aluminium",
        max_thickness=30,
        angle=0
    )
    print(result)

    print("\nSteel Recommendations")
    result = recommend_design(
        desired_v50=800,
        projectile="20MMFSP",
        material_family="Steel",
        max_thickness=30,
        angle=0
    )
    print(result)