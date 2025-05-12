import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set page title
st.set_page_config(page_title="Autism Detection", layout="centered")
st.title("🧠 Autism Detection from Facial Image")

# Load class names
with open("autism_labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Prediction function
def predict_image(image):
    image = image.resize((256, 256))  # Resize to match training input
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    return class_names[predicted_index], confidence

# Upload image
uploaded_file = st.file_uploader("Upload a facial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("⏳ Classifying...")
    label, confidence = predict_image(image)

    st.success(f"✅ Prediction: **{label.upper()}**")
    st.info(f"🔍 Confidence Score: **{confidence}%**")
