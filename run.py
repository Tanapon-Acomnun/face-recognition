# run.py
# RAF-DB Emotion Detection App (Improved Real-World Version)
# Features:
# - Upload image
# - Webcam / Phone camera
# - MediaPipe face detection
# - Face crop before prediction
# - Better real-world performance

import os
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
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Emotion Detection",
    page_icon="😊",
    layout="centered"
)


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_emotion_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model = model.to(device)
    model.eval()

    return model


# =========================
# LOAD MEDIAPIPE
# =========================
@st.cache_resource
def load_face_detector():
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )


model = load_emotion_model()
face_detector = load_face_detector()


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

    # PIL -> OpenCV
    img_np = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = face_detector.process(
        cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    )

    if not results.detections:
        return None, None

    # Use first detected face
    detection = results.detections[0]

    bbox = detection.location_data.relative_bounding_box

    h, w, _ = img_cv.shape

    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))
    width = int(bbox.width * w)
    height = int(bbox.height * h)

    # Add padding
    padding = 20

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + width + padding)
    y2 = min(h, y + height + padding)

    # Crop face
    face_crop = img_np[y1:y2, x1:x2]

    # Draw rectangle for display
    display_img = img_np.copy()

    cv2.rectangle(
        display_img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    return Image.fromarray(face_crop), Image.fromarray(display_img)


# =========================
# PREDICTION
# =========================
def predict_emotion(face_image):

    img_tensor = transform(
        face_image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(img_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities


# =========================
# UI
# =========================
st.title("😊 AI Emotion Detection App")
st.write(
    "This upgraded version detects and crops the face first "
    "for better real-world emotion recognition."
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
# UPLOAD
# =========================
if input_mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)


# =========================
# CAMERA
# =========================
elif input_mode == "Use Camera":

    camera_photo = st.camera_input(
        "Take a picture"
    )

    if camera_photo:
        image = Image.open(camera_photo)


# =========================
# PROCESS
# =========================
if image is not None:

    # Detect face
    face_crop, detected_img = detect_and_crop_face(image)

    if face_crop is None:

        st.error(
            "No face detected. Please upload a clearer face image."
        )

    else:

        # Show detected face box
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
        predicted_class, probabilities = predict_emotion(
            face_crop
        )

        predicted_emotion = emotion_labels[
            predicted_class
        ]

        confidence = (
            probabilities[predicted_class].item() * 100
        )

        # =========================
        # RESULTS
        # =========================
        st.subheader(
            f"Predicted Emotion: {predicted_emotion}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

        # =========================
        # ALL SCORES
        # =========================
        st.write(
            "### Emotion Confidence Scores"
        )

        for i, emotion in enumerate(
            emotion_labels
        ):

            score = probabilities[i].item()

            st.progress(float(score))

            st.write(
                f"{emotion}: {score * 100:.2f}%"
            )