import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Real-Time Face Detection (Powered by GitHub Copilot AI)")

st.write("Upload an image or use your webcam to detect faces.")

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image, faces

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam (beta)"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image.convert('RGB'))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result_img, faces = detect_faces(image_cv)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_column_width=True)

elif option == "Use Webcam (beta)":
    st.warning("Webcam support in Streamlit is experimental. If you don't see a video, try using the image upload option.")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            result_img, faces = detect_faces(frame)
            FRAME_WINDOW.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        cap.release()
