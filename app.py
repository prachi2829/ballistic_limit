import streamlit as st
from src.predict import predict_v50, df
from src.recommender import recommend_design

projectiles = sorted(df["projectile"].unique())
materials = sorted(df["Material"].unique())

st.set_page_config(
    page_title="Ballistic Limit Prediction",
    layout="wide"
)

st.title("Machine Learning-Based Prediction and Design Recommendation System")

st.markdown("""
This application predicts the **Ballistic Limit (V50)** of armor materials
using a trained XGBoost model and recommends suitable armor configurations
based on a desired V50.
""")

tab1, tab2 = st.tabs([
    "V50 Prediction",
    "Design Recommendation"
])


with tab1:

    st.header("Ballistic Limit (V50) Prediction")

    col1, col2 = st.columns(2)

    with col1:
        projectile = st.selectbox(
            "Select Projectile",
            projectiles
        )

        material = st.selectbox(
            "Select Material",
            materials
        )

    with col2:
        thickness = st.slider(
            "Thickness (mm)",
            min_value=1.0,
            max_value=50.0,
            value=25.0,
            step=0.5
        )

        angle = st.slider(
            "Impact Angle (°)",
            min_value=0,
            max_value=90,
            value=0,
            step=5
        )

    if st.button("Predict V50"):

        prediction = predict_v50(
            projectile,
            material,
            thickness,
            angle
        )

        st.success(
            f"Predicted Ballistic Limit (V50): **{prediction} m/s**"
        )

with tab2:

    st.header("Armor Design Recommendation")

    col1, col2 = st.columns(2)

    with col1:

        desired_v50 = st.number_input(
            "Desired V50 (m/s)",
            min_value=100,
            max_value=2000,
            value=800,
            step=10
        )

        projectile = st.selectbox(
            "Projectile",
            projectiles,
            key="rec_projectile"
        )

        material_family = st.selectbox(
            "Material Family",
            ["Any", "Aluminium", "Steel"]
        )

    with col2:

        max_thickness = st.slider(
            "Maximum Thickness (mm)",
            min_value=1.0,
            max_value=50.0,
            value=30.0,
            step=0.5
        )

        angle = st.slider(
            "Impact Angle (°)",
            min_value=0,
            max_value=90,
            value=0,
            step=5,
            key="rec_angle"
        )

    if st.button("Recommend Design"):

        recommendations = recommend_design(
            desired_v50=desired_v50,
            projectile=projectile,
            material_family=material_family,
            max_thickness=max_thickness,
            angle=angle
        )

        st.subheader("Recommended Armor Configurations")

        st.dataframe(
            recommendations,
            use_container_width=True
        )

        if recommendations.iloc[0]["Difference"] > 50:
            st.warning(
                "No configuration closely matches the requested V50. "
                "Consider increasing the maximum thickness or selecting a different material family."
            )





st.divider()

st.caption(
    "Developed as part of the DMSRDE (DRDO) Summer Internship Project | Machine Learning-Based Prediction and Design Recommendation System for Ballistic Limit (V50)"
)