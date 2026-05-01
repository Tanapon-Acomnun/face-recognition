# run.py
# Streamlit Cloud Ready RAF-DB Emotion Detection App
# Features:
# - Hugging Face model auto-download
# - Upload image
# - Webcam / Phone camera
# - MediaPipe face detection
# - Face crop before prediction

import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp


# =========================
# CONFIG
# =========================
MODEL_PATH = "best_emotion_model.pth"

# Hugging Face direct download link
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

        st.info("Downloading model from Hugging Face...")

        response = requests.get(
            MODEL_URL,
            stream=True
        )

        if response.status_code == 200:

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

            st.success(
                "Model downloaded successfully."
            )

        else:

            st.error(
                "Failed to download model from Hugging Face."
            )

            st.stop()


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
# FACE DETECTOR
# =========================
@st.cache_resource
def load_face_detector():

    return face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )


model = load_emotion_model()

face_detector = load_face_detector()


# =========================
# TRANSFORMS
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

    img_cv = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    )

    results = face_detector.process(
        cv2.cvtColor(
            img_cv,
            cv2.COLOR_BGR2RGB
        )
    )

    if not results.detections:
        return None, None

    detection = results.detections[0]

    bbox = (
        detection
        .location_data
        .relative_bounding_box
    )

    h, w, _ = img_cv.shape

    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))

    width = int(bbox.width * w)
    height = int(bbox.height * h)

    # Padding
    padding = 20

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(
        w,
        x + width + padding
    )

    y2 = min(
        h,
        y + height + padding
    )

    face_crop = img_np[
        y1:y2,
        x1:x2
    ]

    display_img = img_np.copy()

    cv2.rectangle(
        display_img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    return (
        Image.fromarray(face_crop),
        Image.fromarray(display_img)
    )


# =========================
# PREDICTION
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
    "Upload an image or use your camera for "
    "real-world face emotion detection."
)


# =========================
# INPUT MODE
# =========================
input_mode = st.radio(
    "Choose Input Method:",
    ["Upload Image", "Use Camera"]
)


image = None


# =========================
# UPLOAD IMAGE
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
# CAMERA
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
# PROCESS
# =========================
if image is not None:

    face_crop, detected_img = (
        detect_and_crop_face(image)
    )

    if face_crop is None:

        st.error(
            "No face detected. "
            "Please upload a clearer image."
        )

    else:

        # Show face box
        st.image(
            detected_img,
            caption="Detected Face",
            use_container_width=True
        )

        # Show crop
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

        # RESULTS
        st.subheader(
            f"Predicted Emotion: "
            f"{predicted_emotion}"
        )

        st.write(
            f"Confidence: "
            f"{confidence:.2f}%"
        )

        # ALL SCORES
        st.write(
            "### Emotion Confidence Scores"
        )

        for i, emotion in enumerate(
            emotion_labels
        ):

            score = probabilities[
                i
            ].item()

            st.progress(
                float(score)
            )

            st.write(
                f"{emotion}: "
                f"{score * 100:.2f}%"
            )
