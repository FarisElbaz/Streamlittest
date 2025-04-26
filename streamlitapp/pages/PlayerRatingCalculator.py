import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import shap
st.title("Player Rating Prediction")

project_dir = Path(__file__).resolve().parent
models_dir = project_dir.parent / "models"
graphs_dir = project_dir.parent / "graphs"
rating_model_path = models_dir / "ridge_model.pkl"
rating_scaler_path = models_dir / "scaler_rating.pkl"
shap_summary_plot_path = graphs_dir / "shap_summary_plot_rating.png"  
shap_values_path = graphs_dir / "shap_values_rating.pkl"

try:
    rating_model = joblib.load(rating_model_path)
    rating_scaler = joblib.load(rating_scaler_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

st.sidebar.header("Player Stats Input")

goals_per_game = st.sidebar.slider("Goals per Game", 0.0, 6.0, 0.5)
assists_per_game = st.sidebar.slider("Assists per Game", 0.0, 6.0, 0.3)
duels_won_per_game = st.sidebar.slider("Total Duels Won per Game", 0.0, 40.0, 5.0)
key_passes_per_game = st.sidebar.slider("Key Passes per Game", 0.0, 15.0, 1.0)
accurate_passes_per_game = st.sidebar.slider("Accurate Passes per Game", 0.0, 100.0, 70.0)
total_duels_won_percentage = st.sidebar.slider("Total Duels Won Percentage", 0.0, 100.0, 50.0)
accurate_passes_percentage = st.sidebar.slider("Accurate Passes Percentage", 0.0, 100.0, 85.0)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Player Stats Summary")
    st.markdown(f"- **Goals/Game**: {goals_per_game}")
    st.markdown(f"- **Assists/Game**: {assists_per_game}")
    st.markdown(f"- **Duels Won/Game**: {duels_won_per_game}")
    st.markdown(f"- **Key Passes/Game**: {key_passes_per_game}")
    st.markdown(f"- **Accurate Passes/Game**: {accurate_passes_per_game}")
    st.markdown(f"- **Duels Won Percentage**: {total_duels_won_percentage}%")
    st.markdown(f"- **Accurate Passes Percentage**: {accurate_passes_percentage}%")

with col2:
    st.subheader("SHAP Summary Plot")
    try:
        shap_values = joblib.load(shap_values_path)

        plt.figure()
        shap.summary_plot(shap_values.values, shap_values.data, feature_names=shap_values.feature_names, show=False)
        st.pyplot(plt.gcf())
        plt.close()
    except FileNotFoundError:
        st.error(f"SHAP values file not found at {shap_values_path}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.button("Predict Player Rating"):
    input_data = pd.DataFrame([{
        'goalsPerGame': goals_per_game,
        'assistsPerGame': assists_per_game,
        'totalDuelsWonPerGame': duels_won_per_game,
        'keyPassesPerGame': key_passes_per_game,
        'accuratePassesPerGame': accurate_passes_per_game,
        'totalDuelsWonPercentage': total_duels_won_percentage,
        'accuratePassesPercentage': accurate_passes_percentage,
    }])

    input_scaled = rating_scaler.transform(input_data)

    rating = rating_model.predict(input_scaled)[0]
    rating_clipped = min(rating, 10)

    st.success(f"**Predicted Player Rating:** {rating_clipped:.2f}")