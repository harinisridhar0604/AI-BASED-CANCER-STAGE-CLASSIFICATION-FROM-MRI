import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ── Model load (once only) ──
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/resnet50_model.h5")

model = load_model()
CLASSES = ["Stage I", "Stage II", "Stage III", "Stage IV"]

# ── Grad-CAM ──
def gradcam(model, img_array):
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer("conv5_block3_out").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        cls = tf.argmax(preds[0])
        loss = preds[:, cls]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return cam.numpy()

# ── UI ──
st.set_page_config(page_title="AI Cancer Classifier", layout="wide")
st.title("AI Based Cancer Stage Classification from MRI")
st.markdown("Upload an MRI image — get stage prediction + Grad-CAM heatmap")

with st.sidebar:
    st.header("Settings")
    show_cam = st.toggle("Show Grad-CAM", value=True)
    threshold = st.slider("Confidence threshold (%)", 50, 99, 75)

uploaded = st.file_uploader("Upload MRI scan", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0).astype("float32")

    with st.spinner("Classifying..."):
        preds = model.predict(input_tensor)

    stage = CLASSES[np.argmax(preds[0])]
    confidence = float(np.max(preds[0])) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Stage", stage)
    col2.metric("Confidence", f"{confidence:.1f}%")
    col3.metric("Model", "ResNet50")

    st.subheader("Stage probabilities")
    for i, cls in enumerate(CLASSES):
        st.progress(float(preds[0][i]), text=f"{cls} — {preds[0][i]*100:.1f}%")

    if show_cam:
        st.subheader("Grad-CAM heatmap")
        cam = gradcam(model, input_tensor)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            np.uint8(img_array * 255), 0.6, heatmap, 0.4, 0
        )
        col_a, col_b = st.columns(2)
        col_a.image(img, caption="Original MRI")
        col_b.image(overlay, caption="Grad-CAM overlay")
