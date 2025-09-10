import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from huggingface_hub import hf_hub_download

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🫁",
    layout="wide"
)

# --- Initialize session state for selected page ---
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "About Pneumonia"

if 'show_pneumonia_advice' not in st.session_state:
    st.session_state.show_pneumonia_advice = False

# --- Load the Model (once) ---
@st.cache_resource
def load_my_model():
    try:
        MODEL_PATH = "my_model.keras"
        # Download the keras model file from HF Hub if not present
        if not os.path.exists(MODEL_PATH):
            # استخدم hf_hub_download لتحميل الملف من الريبو
            local_path = hf_hub_download(
                repo_id="abdelrahmanemam10/Chest_X_Ray_pneumonia_detection",
                filename="my_model.keras",
                # سوف يُخزّن في الكاش المحلي تلقائيًا
            )
            # أحيانًا hf_hub_download يعيد المسار في كاش، مجرد استخدامه مباشر
            # لكن إذا أردت تنقله للمجلد الحالي:
            # os.replace(local_path, MODEL_PATH)
            MODEL_PATH = local_path

        model = keras.models.load_model(MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"Error loading the model: {e}")
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

def navigate_to_page(page_name):
    st.session_state.selected_page = page_name
    st.rerun()

# --- Pages ---
if st.session_state.selected_page == "About Pneumonia":
    st.title("About Pneumonia 🫁")
    st.markdown("""
    Pneumonia is an infection that inflames the air sacs...
    """)
    if st.button("Go to Pneumonia Detector"):
        navigate_to_page("Pneumonia Detector")

elif st.session_state.selected_page == "Pneumonia Detector":
    st.title("AI-Powered Pneumonia Detector 🔬")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.write("Classifying...")

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(opencv_image, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            st.markdown("<h2 style='color: red;'>Prediction: Pneumonia 🔴</h2>", unsafe_allow_html=True)
            st.write(f"Confidence: {prediction[0][0]*100:.2f}%")
            st.session_state.show_pneumonia_advice = True
            st.markdown("""
            <div style="background-color:#ffe0e0;padding:10px;border-radius:5px;">
                <p style="color:red;font-weight:bold;">
                    ⚠️ It appears there is a possibility of pneumonia.<br>
                    Please check the "What to do if Pneumonia" section in the sidebar.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>Prediction: Normal ✅</h2>", unsafe_allow_html=True)
            st.write(f"Confidence: {(1 - prediction[0][0])*100:.2f}%")
            st.session_state.show_pneumonia_advice = False
            st.markdown("""
            <div style="background-color:#e0ffe0;padding:10px;border-radius:5px;">
                <p style="color:green;font-weight:bold;">
                    👍 The image does not show signs of pneumonia.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")
        st.warning("Disclaimer: This AI-based prediction is for informational purposes only and does not substitute professional medical advice.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to About Pneumonia", key="detector_to_about_btn"):
            navigate_to_page("About Pneumonia")
    with col2:
        if st.session_state.show_pneumonia_advice:
            if st.button("Go to What to do if Pneumonia", key="detector_to_advice_btn"):
                navigate_to_page("What to do if Pneumonia")

elif st.session_state.selected_page == "What to do if Pneumonia":
    if st.session_state.show_pneumonia_advice:
        st.title("What to do if you have Pneumonia? 🚨")
        st.markdown("""
        If results indicate a possibility of pneumonia, or if you are experiencing symptoms suggestive of it, it is crucial to take the following actions:
        
        **1. Consult a doctor immediately**  
        **2. Follow doctor’s instructions**  
        **3. Rest, drink fluids, good ventilation, etc.  
        **4. Watch out for emergency warning signs  
        """)
        if st.button("Go to Pneumonia Detector", key="advice_to_detector_btn"):
            navigate_to_page("Pneumonia Detector")
    else:
        st.warning("This section is only available if a potential pneumonia case is detected.")
        if st.button("Go to Pneumonia Detector", key="redirect_to_detector_btn_from_advice"):
            navigate_to_page("Pneumonia Detector")


