import streamlit as st
from PIL import Image
from inference import CaptchaPredictor
import os

# Page config
st.set_page_config(
    page_title="CAPTCHA Recognizer",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    ckpt = os.environ.get("CKPT_PATH", "crnn_best.pth")
    return CaptchaPredictor(ckpt)

predictor = load_model()

# UI
st.title("CAPTCHA Recognizer")
st.markdown("""
Upload a CAPTCHA image and the model will predict the text.
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a CAPTCHA image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded CAPTCHA", use_column_width=True)

    with col2:
        with st.spinner("Predicting..."):
            prediction = predictor.predict(image)

        st.markdown("### Predicted Text")
        st.markdown(
            f"<h1 style='color:#4CAF50; letter-spacing:6px;'>{prediction}</h1>",
            unsafe_allow_html=True
        )