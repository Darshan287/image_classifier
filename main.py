import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
(
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image, ImageOps

# -----------------------------
# Load MobileNetV2 model
# -----------------------------
def load_model():
    try:
        model = MobileNetV2(weights="imagenet")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# -----------------------------
# Preprocess uploaded image
# -----------------------------
def preprocess_image(image):
    img = image.convert("RGB")   # ensure 3 channels
    img = ImageOps.fit(img, (224, 224))  # resize with crop/pad
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Run classification
# -----------------------------
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(
        page_title="AI Image Classifier",
        page_icon="🖼️",
        layout="centered"
    )

    st.title("🖼️ AI Image Classifier")
    st.write("Upload an image and let AI tell you what it sees!")

    @st.cache_resource
    def load_cached_model():
        with st.spinner("Loading AI model... this may take a while ⏳"):
            return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image = Image.open(uploaded_file)

        if st.button("🔍 Classify Image"):
            with st.spinner("Analyzing Image..."):
                predictions = classify_image(model, image)
                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()
