import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

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
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_my_model():
    try:
        # Ensure 'my_model.keras' is in the same directory as app.py
        model = keras.models.load_model("D:\Projects\Deep Learning\ChestX_Ray\my_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}. Please ensure 'my_model.keras' is in the same directory as 'app.py'.")
        st.stop()

model = load_my_model()

# --- Sidebar Navigation ---
page_options = ["About Pneumonia", "Pneumonia Detector"]
if st.session_state.show_pneumonia_advice:
    page_options.append("What to do if Pneumonia")

# Use st.sidebar.radio for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    page_options,
    index=page_options.index(st.session_state.selected_page), # Set initial selection
    key="sidebar_radio"
)

# Update session state based on sidebar selection
st.session_state.selected_page = selected_page

# --- Helper function to navigate pages ---
def navigate_to_page(page_name):
    st.session_state.selected_page = page_name
    st.rerun() # Rerun the app to update the displayed page

# --- Content for each Page ---

# Page 1: About Pneumonia
if st.session_state.selected_page == "About Pneumonia":
    st.title("About Pneumonia 🫁")

    st.markdown("""
    Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing.
    """)

    st.header("Causes of Pneumonia 🦠")
    st.markdown("""
    Pneumonia can be caused by a variety of organisms, including:
    *   **Bacteria:** The most common cause of bacterial pneumonia.
    *   **Viruses:** Such as influenza viruses, respiratory syncytial virus (RSV), and coronaviruses (COVID-19).
    *   **Fungi:** Less common, usually affecting people with weakened immune systems.
    """)

    st.header("Symptoms of Pneumonia 🤒")
    st.markdown("""
    Pneumonia symptoms range from mild to severe and may include:
    *   Cough, which may produce phlegm (mucus) that is green, yellow, or bloody.
    *   Fever, sweating, and shaking chills.
    *   Shortness of breath.
    *   Chest pain that worsens when you breathe deeply or cough.
    *   Fatigue and tiredness.
    *   Nausea, vomiting, or diarrhea.
    """)

    st.header("How to Prevent Pneumonia 🛡️")
    st.markdown("""
    Several steps can be taken to help prevent pneumonia:
    *   **Vaccinations:** Get your annual flu shot and the pneumococcal vaccine regularly, especially if you are in a high-risk group.
    *   **Good Hygiene:** Wash your hands regularly with soap and water, or use an alcohol-based hand sanitizer.
    *   **Avoid Smoking:** Smoking damages your lungs' ability to fight off infections.
    *   **Boost Your Immune System:** Eat a healthy diet, get enough sleep, and exercise regularly.
    *   **Avoid Close Contact:** Stay away from sick people as much as possible.
    *   **Cover Your Mouth and Nose:** When you cough or sneeze, use a tissue or your elbow.
    """)

    st.info("This information is for educational purposes only and does not substitute professional medical advice.")

    # Navigation button
    st.markdown("---")
    if st.button("Go to Pneumonia Detector", key="about_to_detector_btn"):
        navigate_to_page("Pneumonia Detector")

# Page 2: Pneumonia Detector
elif st.session_state.selected_page == "Pneumonia Detector":
    st.title("AI-Powered Pneumonia Detector 🔬")
    st.write("Upload a chest X-ray image to determine if it shows signs of pneumonia.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image (smaller size)
        st.image(uploaded_file, caption="Uploaded Image", width=300) # Set width to 300 pixels
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for the model
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) # Convert to RGB

        # Resize the image to the target size (224x224)
        img_resized = cv2.resize(opencv_image, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Rescale to [0, 1]

        # Make prediction
        prediction = model.predict(img_array)

        # Assuming binary classification where 0 is Normal and 1 is Pneumonia
        if prediction[0][0] > 0.5:
            st.markdown("<h2 style='color: red;'>Prediction: Pneumonia 🔴</h2>", unsafe_allow_html=True)
            st.write(f"Confidence: {prediction[0][0]*100:.2f}%")
            st.session_state.show_pneumonia_advice = True # Set flag to show the third page
            st.markdown("""
            <div style="background-color:#ffe0e0;padding:10px;border-radius:5px;">
                <p style="color:red;font-weight:bold;">
                    ⚠️ It appears there is a possibility of pneumonia.
                    <br>
                    Please check the "What to do if Pneumonia" section in the sidebar for guidance.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>Prediction: Normal ✅</h2>", unsafe_allow_html=True)
            st.write(f"Confidence: {(1 - prediction[0][0])*100:.2f}%")
            st.session_state.show_pneumonia_advice = False # Hide the third page
            st.markdown("""
            <div style="background-color:#e0ffe0;padding:10px;border-radius:5px;">
                <p style="color:green;font-weight:bold;">
                    👍 The image does not show signs of pneumonia.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")
        st.warning("Disclaimer: This AI-based prediction is for informational purposes only and does not substitute professional medical advice or diagnosis.")

    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to About Pneumonia", key="detector_to_about_btn"):
            navigate_to_page("About Pneumonia")
    with col2:
        # Only show the button to the advice page if advice is needed
        if st.session_state.show_pneumonia_advice:
            if st.button("Go to What to do if Pneumonia", key="detector_to_advice_btn"):
                navigate_to_page("What to do if Pneumonia")

# Page 3: What to do if Pneumonia (Conditional Display)
elif st.session_state.selected_page == "What to do if Pneumonia":
    # This page should only be accessible if show_pneumonia_advice is True
    if st.session_state.show_pneumonia_advice:
        st.title("What to do if you have Pneumonia? 🚨")

        st.markdown("""
        If results indicate a possibility of pneumonia, or if you are experiencing symptoms suggestive of it, it is crucial to take the following actions immediately:
        """)

        st.header("1. Consult a Doctor Immediately 👨‍⚕️")
        st.markdown("""
        *   **Do not delay:** Pneumonia can be serious, especially for the elderly, young children, and individuals with weakened immune systems.
        *   **Accurate Diagnosis:** Only a doctor can definitively diagnose pneumonia through physical examination, chest X-rays, and possibly blood or sputum tests.
        *   **Appropriate Treatment:** The doctor will prescribe the correct treatment based on the cause of the pneumonia (antibiotics for bacteria, antivirals for viruses, or antifungals).
        """)

        st.header("2. Follow Doctor's Instructions Carefully 💊")
        st.markdown("""
        *   **Complete the full course of medication:** Even if you feel better, do not stop taking the prescribed medications.
        *   **Rest:** Get plenty of rest to help your body recover.
        *   **Fluids:** Drink plenty of fluids to prevent dehydration and help loosen mucus.
        *   **Avoid Irritants:** Stay away from smoking, secondhand smoke, and air pollution.
        """)

        st.header("3. Self-Care and Support 🏡")
        st.markdown("""
        *   **Manage Fever:** Use over-the-counter fever reducers (like paracetamol or ibuprofen) as directed by your doctor.
        *   **Relieve Cough:** Some cough suppressants may be used after consulting your doctor, but coughing is important for clearing phlegm.
        *   **Good Ventilation:** Ensure the room you are in well-ventilated.
        *   **Prevent Spread:** Wash your hands regularly and cover your mouth when coughing or sneezing.
        """)

        st.header("4. Emergency Warning Signs 🚨")
        st.markdown("""
        Seek emergency medical help if you experience any of the following symptoms:
        *   Severe shortness of breath or difficulty breathing.
        *   Severe chest pain.
        *   Bluish discoloration of the lips or fingertips.
        *   Confusion or changes in awareness.
        *   Very high fever that does not respond to treatment.
        """)

        st.warning("These guidelines are general advice and do not replace specialized medical consultation. Your health is paramount; do not hesitate to seek medical help.")

        # Navigation button
        st.markdown("---")
        if st.button("Go to Pneumonia Detector", key="advice_to_detector_btn"):
            navigate_to_page("Pneumonia Detector")
    else: # <--- هذا هو السطر الذي كان يسبب المشكلة في المسافات البادئة
        # If user tries to access this page directly without a positive detection
        st.warning("This section is only available if a potential pneumonia case is detected.")
        st.info("Please go to the 'Pneumonia Detector' section to upload an image.")
        if st.button("Go to Pneumonia Detector", key="redirect_to_detector_btn_from_advice"):
            navigate_to_page("Pneumonia Detector")

