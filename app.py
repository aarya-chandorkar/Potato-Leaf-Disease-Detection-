import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2

# --- Page Config ---
st.set_page_config(page_title="Potato Disease Classifier", 
                   page_icon="🥔", 
                   layout="wide")

# --- Load Model ---
MODEL_PATH = "potatoes.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define Class Names
CLASS_NAMES = ["Potato_Early_blight", "Potato_Late_blight", "Potato_healthy"]

# --- Sidebar Navigation ---
st.sidebar.title("🥔 Potato Disease App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📤 Upload & Predict", "ℹ About"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("🌿 Potato Disease Classification")
    st.markdown("""
    This application uses a *Convolutional Neural Network (CNN)* trained on the 
    [PlantVillage Dataset](https://www.kaggle.com/arjuntejaswi/plant-village) to classify 
    potato leaf diseases into three categories:
    - *Early Blight*  
    - *Late Blight*  
    - *Healthy*
    """)
    st.image("https://www.researchgate.net/publication/331228821/figure/fig1/AS:725327504080896@1550735745921/Potato-leaves.jpg",
             caption="Potato Leaf Diseases", use_column_width=True)

# --- Upload & Predict Page ---
elif page == "📤 Upload & Predict":
    st.title("🔎 Upload an Image for Classification")

    uploaded_file = st.file_uploader("Upload a potato leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load & preprocess
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image.resize((256, 256))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predictions
        preds = model.predict(img_array)[0]
        top_idxs = np.argsort(preds)[::-1]

        # --- Show Top Prediction ---
        st.success(f"✅ Prediction: *{CLASS_NAMES[top_idxs[0]]}*")
        st.info(f"🔍 Confidence: {preds[top_idxs[0]]*100:.2f}%")

        # --- Bar Chart for All Predictions ---
        st.subheader("📊 Prediction Probabilities")
        fig = px.bar(
            x=[CLASS_NAMES[i] for i in top_idxs],
            y=[preds[i] for i in top_idxs],
            labels={"x": "Class", "y": "Confidence"},
            text=[f"{preds[i]*100:.2f}%" for i in top_idxs]
        )
        st.plotly_chart(fig)

        # --- Grad-CAM Heatmap ---
        st.subheader("🔥 Model Attention (Grad-CAM)")
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(model.layers[-3].name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, np.argmax(preds)]
        grads = tape.gradient(loss, conv_outputs)[0]
        guided_grads = grads.numpy()
        conv_outputs = conv_outputs[0].numpy()
        weights = np.mean(guided_grads, axis=(0, 1))
        cam = np.dot(conv_outputs, weights)
        cam = cv2.resize(cam, (256, 256))
        cam = np.maximum(cam, 0)
        heatmap = cam / cam.max()
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(np.array(image.resize((256, 256))), 0.6, heatmap, 0.4, 0)

        st.image(superimposed, caption="Grad-CAM Heatmap", use_column_width=True)

# --- About Page ---
elif page == "ℹ About":
    st.title("📌 About this Project")
    st.markdown("""
    - *Author*: Aarya Chandorkar  
    - *Model*: Convolutional Neural Network (CNN)  
    - *Dataset*: [PlantVillage - Potato Disease subset](https://www.kaggle.com/arjuntejaswi/plant-village)  
    - *Accuracy*: ~95% on test set  
    - *Tech Stack*: Python, TensorFlow/Keras, Streamlit, Plotly  
    - *Features*: Image Upload, Disease Prediction, Confidence Scores, Grad-CAM Visualization  
    """)