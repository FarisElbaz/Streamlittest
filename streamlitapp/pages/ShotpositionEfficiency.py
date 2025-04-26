import streamlit as st
from pathlib import Path
from PIL import Image

st.title("Goal Scoring Efficiency Analysis")

st.markdown("""
### Key Findings
- This analysis focuses on the efficiency of goal scoring in football, comparing the number of goals scored to the number of shots taken from two bins, inside or outside of the box.
- The first graph shows the efficiency of goal scoring, indicating how many goals were scored per shot taken per bin. Inside the box shooing has a 3x higher efficiency than outside the box shooting.
- The second graph is a heatmap showing the density of shots taken from different areas of the pitch, with a focus on the number of goals scored and missed shots.
- The heatmap indicates that the majority of goals are scored from inside the box, close to the centeral penalty area, with a significant number of missed shots occurring outside this area.
""")
project_dir = Path(__file__).resolve().parent
graphs_dir = project_dir.parent / "graphs"

image_files = [
    graphs_dir / "goal_scoring_efficiency.png",
    graphs_dir / "shot_density_goals_vs_misses_non_scaled.png"
]

if not any(image_file.exists() for image_file in image_files):
    st.warning("No relevant images found in the graphs directory.")
else:
    st.subheader("Goal Scoring Efficiency and Shot Density Graphs")
    for image_file in image_files:
        if image_file.exists():
            st.markdown(f"### {image_file.stem.replace('_', ' ').title()}")
            image = Image.open(image_file)
            st.image(image, caption=image_file.stem.replace('_', ' ').title(), use_column_width=True)
        else:
            st.warning(f"Image not found: {image_file.name}")