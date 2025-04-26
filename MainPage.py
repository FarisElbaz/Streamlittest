import streamlit as st

st.set_page_config(page_title="Football Analytics App", layout="wide")

st.title("Welcome to the Football Analytics App! ⚽")
st.write("Use the sidebar to navigate between Goal the different models.")

#ignore this
import numpy as np

class ClippedModel:
    def __init__(self, model, max_value=10):
        self.model = model
        self.max_value = max_value

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.clip(predictions, None, self.max_value)