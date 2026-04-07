import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

@st.cache_resource
def load_model_file():
    model = load_model("tomato_disease_classifier.h5")
    return model

model = load_model_file()

class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

st.set_page_config(page_title="Tomato Disease Classifier", layout="centered")

st.title("🍅 Tomato Disease Classification")
st.markdown("### Upload a tomato leaf image to detect disease")

st.write("This model uses Deep Learning (CNN) to classify plant diseases.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    probs = prediction[0]

    predicted_index = np.argmax(probs)
    predicted_class = class_names[predicted_index]
    confidence = probs[predicted_index]

    st.subheader("🔍 Prediction Result")
    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    st.subheader("📊 Top 3 Predictions")

    top3_idx = np.argsort(probs)[-3:][::-1]

    for i in top3_idx:
        st.write(f"{class_names[i]}: {probs[i]:.2f}")

    st.subheader("💡 Recommendation")

    if predicted_class == "Healthy":
        st.success("✅ Your plant looks healthy. Keep maintaining good care!")
    else:
        st.warning("⚠️ Disease detected. Consider proper treatment and monitoring.")


st.markdown("---")

st.subheader("📘 About the Model")
st.write("This model is trained using a Convolutional Neural Network (CNN) on tomato leaf images.")
st.write("It classifies images into multiple disease categories.")

st.info("⚠️ This tool is for educational purposes only and not a substitute for expert agricultural advice.")


st.markdown("---")
st.write("Built using Deep Learning + Streamlit 🚀")
