# run.py
# Streamlit Cloud Ready RAF-DB Emotion Detection App
# Uses OpenCV Haar Cascade (lighter + deployment-friendly)
# Features:
# - Hugging Face model auto-download
# - Upload image
# - Webcam / Phone camera
# - Face detection
# - Emotion prediction

import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2


# =========================
# CONFIG
# =========================
MODEL_PATH = "best_emotion_model.pth"

MODEL_URL = (
    "https://huggingface.co/SoftSkinz/face-detector/resolve/main/"
    "best_emotion_model.pth"
)

IMG_SIZE = 224
NUM_CLASSES = 7

emotion_labels = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(
    page_title="AI Emotion Detection",
    page_icon="😊",
    layout="centered"
)

st.sidebar.info(
    "RAF-DB Emotion Detection | Hugging Face + Streamlit Cloud"
)


# =========================
# DEVICE
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# DOWNLOAD MODEL
# =========================
def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading model from Hugging Face..."):

            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=120
            )

            if response.status_code != 200:
                st.error(
                    "Failed to download model from Hugging Face."
                )
                st.stop()

            total_size = int(
                response.headers.get(
                    "content-length",
                    0
                )
            )

            progress_bar = st.progress(0)

            downloaded_size = 0

            with open(MODEL_PATH, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):

                    if chunk:

                        f.write(chunk)

                        downloaded_size += len(chunk)

                        if total_size > 0:

                            progress = int(
                                (downloaded_size / total_size) * 100
                            )

                            progress_bar.progress(
                                min(progress, 100)
                            )

            progress_bar.empty()

            st.success(
                "Model downloaded successfully."
            )


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_emotion_model():

    download_model()

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


# =========================
# LOAD FACE DETECTOR
# =========================
@st.cache_resource
def load_face_detector():

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    return detector


# =========================
# LOAD RESOURCES
# =========================
model = load_emotion_model()

face_detector = load_face_detector()


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# FACE DETECTION + CROP
# =========================
def detect_and_crop_face(image):

    img_np = np.array(
        image.convert("RGB")
    )

    # Convert for OpenCV
    img_bgr = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast for darker / blurry webcam images
    gray = cv2.equalizeHist(gray)

    # Multi-scale detection
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,      # More sensitive
        minNeighbors=3,        # Easier detection
        minSize=(30, 30),      # Smaller faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Retry on larger resized image if no face found
    if len(faces) == 0:

        enlarged = cv2.resize(
            gray,
            None,
            fx=1.5,
            fy=1.5
        )

        retry_faces = face_detector.detectMultiScale(
            enlarged,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(retry_faces) > 0:

            # Scale coordinates back
            faces = []

            for (x, y, w, h) in retry_faces:

                faces.append((
                    int(x / 1.5),
                    int(y / 1.5),
                    int(w / 1.5),
                    int(h / 1.5)
                ))

    if len(faces) == 0:
        return None, None

    # Choose largest detected face
    faces = sorted(
        faces,
        key=lambda f: f[2] * f[3],
        reverse=True
    )

    x, y, w, h = faces[0]

    # More generous padding
    padding_x = int(w * 0.25)
    padding_y = int(h * 0.35)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(
        img_np.shape[1],
        x + w + padding_x
    )

    y2 = min(
        img_np.shape[0],
        y + h + padding_y
    )

    if x2 <= x1 or y2 <= y1:
        return None, None

    # Crop face
    face_crop = img_np[
        y1:y2,
        x1:x2
    ]

    # Draw box
    display_img = img_np.copy()

    cv2.rectangle(
        display_img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    return (
        Image.fromarray(face_crop),
        Image.fromarray(display_img)
    )


# =========================
# PREDICT EMOTION
# =========================
def predict_emotion(face_image):

    img_tensor = transform(
        face_image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(img_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

    return predicted_class, probabilities


# =========================
# UI
# =========================
st.title("😊 AI Emotion Detection App")

st.write(
    "Upload an image or use your camera "
    "to detect emotions in real time."
)


# =========================
# INPUT METHOD
# =========================
input_mode = st.radio(
    "Choose Input Method:",
    ["Upload Image", "Use Camera"]
)

image = None


# =========================
# FILE UPLOAD
# =========================
if input_mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )


# =========================
# CAMERA INPUT
# =========================
elif input_mode == "Use Camera":

    camera_photo = st.camera_input(
        "Take a picture"
    )

    if camera_photo:

        image = Image.open(
            camera_photo
        )


# =========================
# PROCESS IMAGE
# =========================
if image is not None:

    with st.spinner(
        "Detecting face and analyzing emotion..."
    ):

        face_crop, detected_img = (
            detect_and_crop_face(image)
        )

    if face_crop is None:

        st.error(
            "No face detected. "
            "Please use a clearer face image."
        )

    else:

        # Show original with box
        st.image(
            detected_img,
            caption="Detected Face",
            use_container_width=True
        )

        # Show cropped face
        st.image(
            face_crop,
            caption="Face Crop Used for Prediction",
            use_container_width=True
        )

        # Predict
        predicted_class, probabilities = (
            predict_emotion(face_crop)
        )

        predicted_emotion = (
            emotion_labels[predicted_class]
        )

        confidence = (
            probabilities[
                predicted_class
            ].item() * 100
        )

        # MAIN RESULT
        st.subheader(
            f"Predicted Emotion: {predicted_emotion}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

        # ALL SCORES
        st.write(
            "### Emotion Confidence Scores"
        )

        for i, emotion in enumerate(
            emotion_labels
        ):

            score = probabilities[i].item()

            st.write(
                f"{emotion}: {score * 100:.2f}%"
            )

            st.progress(
                float(score)
            )
