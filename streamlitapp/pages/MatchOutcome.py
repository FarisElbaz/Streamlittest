import streamlit as st
from pathlib import Path
from PIL import Image

st.title("Match Outcome Analysis")

st.markdown("""
### Key Findings
- Match outcomes vary significantly across leagues and seasons.
- The data highlights trends in win, draw, and loss rates for teams across different leagues.
- Certain leagues exhibit higher win rates for home teams, while others show more balanced outcomes. See set 2: England, the only team to pass the 15000 mark.
- The analysis includes data from 223 different leagues, with 70 leagues in each set.
- These insights can help identify patterns and anomalies in match results.
""")

project_dir = Path(__file__).resolve().parent
graphs_dir = project_dir.parent / "graphs"

image_files = sorted(graphs_dir.glob("graph_set_*.png"))

if not image_files:
    st.warning("No images found matching the pattern 'graph_set_x.png'.")
else:
    st.subheader("Match Outcome Graphs across leagues")
    for image_file in image_files:
        st.markdown(f"### {image_file.stem.replace('_', ' ').title()}")
        image = Image.open(image_file)
        st.image(image, caption=image_file.stem.replace('_', ' ').title(), use_container_width=True)
