import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Goal Probability Prediction")
st.markdown("""Predict the probability of scoring a goal based on various features.
- This model is trained on a dataset of football shots scraped from Fotmob.com
- It contains data from the last 5 seasons for every player available in the Fotmob database
- The model predicts is a LightGBM model that predicts the probability of scoring a goal based on the following features:
    - X and Y coordinates of the shot
    - Distance from the goal
    - Play situation (e.g., Regular Play, Penalty, Free Kick)
    - Shot type (e.g., Left Foot, Right Foot, Other Body Parts)
    - Whether the shot is from inside the box
    - Whether the shot is a central shot
    - Whether the player is on the home team
    - Game time bin (e.g., early, mid, late)
    - Fatigue indicator
    - Score difference at the time of the shot
- The model uses SHAP values to explain its predictions. The most impactful feature is the x coordinate, which shows both the most positive and most negative SHAP values.
- This means that a player's position along the field (left to right) strongly affects their likelihood of scoring — certain positions make goals much more or much less likely.
            """)

GOAL_X = 100.0
GOAL_Y = 34.0

project_dir = Path(__file__).resolve().parent
models_dir = project_dir.parent / "models"
model_path = models_dir / "lgb_model.pkl"
scaler_path = models_dir / "scaler1.pkl"
graphs_dir = project_dir.parent / "graphs"
shap_map_path = graphs_dir/ "shap_summary_plot_best_lgb_model.png"


try:
    lgb_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

def calculate_distance(x, y):
    return round(math.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2), 2)

def calculate_angle(x, y):
    dx = GOAL_X - x
    if dx == 0:
        return 0.0
    return round(math.degrees(math.atan((7.32 / 2) / dx)), 2)

# UI inputs
st.sidebar.header("Input Controls")

situation = st.sidebar.selectbox("Situation", [
    "RegularPlay", "Penalty", "FreeKick", "FromCorner",
    "IndividualPlay", "SetPiece", "ThrowInSetPiece"
])
central_shot = st.sidebar.checkbox("Central Shot")
is_home_team = st.sidebar.checkbox("Home Team?")
shot_type = st.sidebar.selectbox("Shot Type", ["LeftFoot", "RightFoot", "OtherBodyParts"])
game_time_bin = st.sidebar.selectbox("Game Phase", ["early", "mid", "late"])

x = st.sidebar.slider("X Coordinate", 0.0, 105.0, 88.0)
y = st.sidebar.slider("Y Coordinate", 0.0, 68.0, 34.0)

# Adjustments for penalty or central shot
if situation == "Penalty":
    x, y = 88.0, 34.0
elif central_shot:
    y = 34.0

# More inputs
fatigue_indicator = st.sidebar.slider("Fatigue Indicator", 0.0, 1.0, 0.5)
score_diff_at_shot = st.sidebar.slider("Score Difference", -5, 5, 0)

# Derived features
distance = calculate_distance(x, y)
angle = calculate_angle(x, y)

# Inputs summary
st.subheader("Features Summary")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"- **X**: {x}")
    st.markdown(f"- **Y**: {y}")
    st.markdown(f"- **Distance to goal**: {distance}")
    st.markdown(f"- **Angle to goal**: {angle}")

with col2:
    st.subheader("SHAP Summary Plot")
    try:
        shap_image = Image.open(shap_map_path)
        st.image(shap_image, caption="SHAP Summary Plot", use_container_width=True)
    except FileNotFoundError:
        st.error("SHAP summary plot not found. Please ensure the file exists.")

# Pitch visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pitch_with_shot(x, y):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw pitch
    plt.plot([0, 0], [0, 68], color="black")
    plt.plot([0, 105], [68, 68], color="black")
    plt.plot([105, 105], [68, 0], color="black")
    plt.plot([105, 0], [0, 0], color="black")
    plt.plot([52.5, 52.5], [0, 68], color="black")

    plt.plot([16.5, 16.5], [54.8, 13.2], color="black")
    plt.plot([0, 16.5], [54.8, 54.8], color="black")
    plt.plot([0, 16.5], [13.2, 13.2], color="black")

    plt.plot([105, 88.5], [54.8, 54.8], color="black")
    plt.plot([88.5, 88.5], [54.8, 13.2], color="black")
    plt.plot([88.5, 105], [13.2, 13.2], color="black")

    plt.plot([0, 5.5], [43.8, 43.8], color="black")
    plt.plot([5.5, 5.5], [43.8, 24.2], color="black")
    plt.plot([0, 5.5], [24.2, 24.2], color="black")

    plt.plot([105, 99.5], [43.8, 43.8], color="black")
    plt.plot([99.5, 99.5], [43.8, 24.2], color="black")
    plt.plot([105, 99.5], [24.2, 24.2], color="black")

    centre_circle = plt.Circle((52.5, 34), 9.15, color="black", fill=False)
    centre_spot = plt.Circle((52.5, 34), 0.5, color="black")
    ax.add_patch(centre_circle)
    ax.add_patch(centre_spot)

    left_pen = plt.Circle((11, 34), 0.5, color="black")
    right_pen = plt.Circle((94, 34), 0.5, color="black")
    ax.add_patch(left_pen)
    ax.add_patch(right_pen)

    # Shot point
    ax.scatter(x, y, color='red', s=100, edgecolors='black', zorder=5)
    ax.text(x + 1, y + 1, f"({x:.1f}, {y:.1f})", color='red', fontsize=12)

    # Arrow toward goal center
    goal_x, goal_y = 105, 34
    dx = goal_x - x
    dy = goal_y - y
    distance = np.sqrt(dx**2 + dy**2)

    arrow_length_scale = 0.25  # You can tweak this scale
    ax.arrow(x, y, dx * arrow_length_scale, dy * arrow_length_scale,
             head_width=1.5, head_length=2.5, fc='red', ec='red')

    # Axes
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect('equal')
    ax.axis('off')

    st.pyplot(fig)


# Prediction
if st.button("Predict"):
    input_dict = {
        'x': x,
        'y': y,
        'distance_from_goal': distance,
        'fatigue_indicator': fatigue_indicator,
        'score_diff_at_shot': score_diff_at_shot,
        'shotType_LeftFoot': int(shot_type == "LeftFoot"),
        'shotType_RightFoot': int(shot_type == "RightFoot"),
        'shotType_OtherBodyParts': int(shot_type == "OtherBodyParts"),
        'situation_RegularPlay': int(situation == "RegularPlay"),
        'situation_Penalty': int(situation == "Penalty"),
        'situation_FreeKick': int(situation == "FreeKick"),
        'situation_FromCorner': int(situation == "FromCorner"),
        'situation_IndividualPlay': int(situation == "IndividualPlay"),
        'situation_SetPiece': int(situation == "SetPiece"),
        'situation_ThrowInSetPiece': int(situation == "ThrowInSetPiece"),
        'isFromInsideBox_True': int(x > 85 and 15.5 <= y <= 52.5),
        'isCentralShot': int(central_shot),
        'isHomeTeam': int(is_home_team),
        'game_time_bin_mid': int(game_time_bin == "mid"),
        'game_time_bin_late': int(game_time_bin == "late"),
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure only the columns the scaler expects are scaled
    numerical_cols = ['x', 'y', 'distance_from_goal', 'fatigue_indicator', 'score_diff_at_shot']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    # Reorder input_df columns to match the model's feature order
    input_df = input_df[lgb_model.feature_name_]

    # Prediction
    proba = lgb_model.predict_proba(input_df)[0]
    pred_class = lgb_model.predict(input_df)[0]

    # Output
    st.success(f"**Goal Probability:** {proba[1]:.2%}")
    st.info(f"**Prediction (1=Goal, 0=No Goal):** {pred_class}")

    # Visualize pitch with the shot
    st.subheader("Shot Location on Pitch")
    draw_pitch_with_shot(x, y)