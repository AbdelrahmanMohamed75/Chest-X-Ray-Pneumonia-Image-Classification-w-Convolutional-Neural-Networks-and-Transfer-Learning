import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from huggingface_hub import hf_hub_download  # جديد

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🫁",
    layout="wide"
)

# --- Initialize session state for selected page ---
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "About Pneumonia" # Start with the first page

# --- Initialize session state for showing pneumonia advice page ---
if 'show_pneumonia_advice' not in st.session_state:
    st.session_state.show_pneumonia_advice = False # Initially hide the advice page

# --- Load the Model (Load once at the beginning) ---
@st.cache_resource  # Cache the model to avoid reloading on every rerun
def load_my_model():
    try:
        # نزّل الموديل من Hugging Face Hub
        model_path = hf_hub_download(
            repo_id="abdelrahmanemam10/Chest_X_Ray_pneumonia_detection",
            filename="my_model.keras"
        )
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}. Please check Hugging Face repo or internet connection.")
        st.stop()

model = load_my_model()

# --- Sidebar Navigation ---
page_options = ["About Pneumonia", "Pneumonia Detector"]
if st.session_state.show_pneumonia_advice:
    page_options.append("What to do if Pneumonia")

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    page_options,
    index=page_options.index(st.session_state.selected_page),
    key="sidebar_radio"
)
st.session_state.selected_page = selected_page

# --- Helper function to navigate pages ---
def navigate_to_page(page_name):
    st.session_state.selected_page = page_name
    st.rerun()

# -------------- باقي الكود كما هو (About Pneumonia, Detector, Advice) --------------
# 👇 انسخ بالظبط كل الصفحات اللي عندك من الكود السابق بدون تغيير

