import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# === Konfigurasi halaman ===
st.set_page_config(page_title="Garbage Classifier", page_icon="♻️", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>♻️ Garbage Classifier - EfNetB2</h1>", unsafe_allow_html=True)
st.write("Real-time prediction using webcam or image upload.")

# === Load model ===
@st.cache_resource
def load_model():
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(6, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.load_weights("model/best_model_weights_trial1.keras")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# === Preprocessing untuk EfficientNet ===
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32')
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# === Fungsi prediksi dan anotasi ===
def predict_and_draw(frame):
    input_tensor = preprocess(frame)
    pred = model.predict(input_tensor, verbose=0)[0]
    class_idx = np.argmax(pred)
    label = labels[class_idx]
    confidence = pred[class_idx]

    h, w, _ = frame.shape
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 255, 0), 3)

    text = f"{label} ({confidence * 100:.2f}%)"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (10, 10), (10 + text_w + 20, 10 + text_h + 20), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 30 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# === Webcam Stream (gunakan tombol kontrol) ===
st.markdown("## 📷 Webcam Stream (Opsional)")

use_webcam = st.checkbox("Aktifkan Webcam Stream")
start_cam = st.button("Mulai Prediksi") if use_webcam else False
stop_cam = st.button("Stop")

if use_webcam and start_cam and not stop_cam:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ Tidak bisa mengakses kamera.")
            break

        frame = cv2.flip(frame, 1)
        result = predict_and_draw(frame)
        stframe.image(result, channels="BGR")

        if stop_cam:
            break

    cap.release()
    cv2.destroyAllWindows()

# === Upload Gambar Alternatif ===
st.markdown("---")
st.markdown("### 🖼️ Atau Upload Gambar:")

uploaded_file = st.file_uploader("Upload file gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    frame = np.array(pil_img)
    result = predict_and_draw(frame)
    st.image(result, channels="BGR")
