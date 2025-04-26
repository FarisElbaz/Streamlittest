import streamlit as st
from pathlib import Path
from PIL import Image

st.title("Home Advantage Effect")
st.markdown("""
### Key Findings
- Teams playing at home tend to have a higher win rate compared to away games.
- The data suggests that crowd support and familiarity with the home ground contribute significantly to performance.
- However, the advantage varies across leagues and seasons, as shown in the graphs below.
- The analysis includes data from 223 different leagues, with 70 leagues in each set,
- The overwhelming majority of leagues show a positive home advantage effect, with home teams winning close to 40% of all matches.
- Additionally, the home advantage effect is more pronounced in leagues with a higher number of matches played as seen in the England and Aregentina leagues.
""")

project_dir = Path(__file__).resolve().parent
graphs_dir = project_dir.parent / "graphs"

image_files = sorted(graphs_dir.glob("home_match_outcomes_set_*.png"))

if not image_files:
    st.warning("No images found matching the pattern 'home_match_outcome_set_x.png'.")
else:
    st.subheader("Home Match Outcome Graphs")
    for image_file in image_files:
        st.markdown(f"### {image_file.stem.replace('_', ' ').title()}")
        image = Image.open(image_file)
        st.image(image, caption=image_file.stem, use_column_width=True)