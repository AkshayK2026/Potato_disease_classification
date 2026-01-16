import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Potato Disease Detector", layout="centered")

st.title("🥔 Potato Leaf Disease Classification")
st.write("Upload a potato leaf image to detect disease")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            response = requests.post(
                API_URL,
                files={"file": ("image.png", img_bytes, "image/png")}
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"🌱 Disease: **{result['disease']}**")
                st.info(f"🔍 Confidence: **{result['confidence'] * 100:.2f}%**")
            else:
                st.error("❌ Failed to get prediction from API")
