import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import sounddevice as sd
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AI Multi-Feature App", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Face Recognition", "Hand Gesture Movement", "Speech to Text", "Image Recognition"],
        icons=["person-square", "hand-index-thumb", "mic", "image"],
        menu_icon="cast",
        default_index=0,
    )

st.title("Real-Time Face Detection and Hand Gesture Detection (Powered by GitHub Copilot AI)")

st.write("Upload an image or use your webcam to detect faces and hand gestures.")

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image, faces

def plot_waveform(audio_data, sample_rate):
    plt.figure(figsize=(6, 2))
    plt.plot(audio_data)
    plt.title('Audio Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def record_audio(duration=10, fs=16000):
    st.info(f"Recording for up to {duration} seconds. Click 'Stop Listening' to end early.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    stop = False
    stop_button = st.button("Stop Listening")
    for i in range(duration * 10):
        sd.sleep(100)
        if stop_button:
            stop = True
            break
    sd.stop()
    if stop:
        st.warning("Stopped listening early.")
    return audio[:i*fs//10] if stop else audio

# --- Face Recognition (with MediaPipe for webcam) ---
if selected == "Face Recognition":
    st.header("Face Recognition")
    option = st.radio("Choose input method:", ("Upload Image", "Use Webcam (MediaPipe)"))
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            result_img, faces = detect_faces(image_cv)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_column_width=True)
    elif option == "Use Webcam (MediaPipe)":
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        cap = None
        if run:
            cap = cv2.VideoCapture(0)
            with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.write("Failed to grab frame")
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)
                    if results.detections:
                        for detection in results.detections:
                            mp_drawing.draw_detection(frame, detection)
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

# --- Hand Gesture Movement (MediaPipe) ---
elif selected == "Hand Gesture Movement":
    st.header("Hand Gesture Detection (Webcam)")
    st.warning("Webcam support in Streamlit is experimental. If you don't see a video, try using the image upload option.")
    run = st.checkbox('Start Hand Gesture Detection')
    FRAME_WINDOW = st.image([])
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    def count_fingers(hand_landmarks, hand_label):
        finger_tips = [4, 8, 12, 16, 20]
        finger_pip = [3, 6, 10, 14, 18]
        count = 0
        if hand_label == 'Right':
            if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pip[0]].x:
                count += 1
        else:
            if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pip[0]].x:
                count += 1
        for tip, pip in zip(finger_tips[1:], finger_pip[1:]):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                count += 1
        return count
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                total_fingers = 0
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        count = count_fingers(hand_landmarks, label)
                        total_fingers += count
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, f'Total Fingers: {total_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(frame, 'Show your hands to the camera', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

# --- Speech to Text (fix error, use sounddevice fallback if needed) ---
elif selected == "Speech to Text":
    st.header("Speech to Text Recognition")
    st.write("Click 'Start Listening' and speak. The waveform of your speech will be shown. Click 'Stop Listening' to end early.")
    import speech_recognition as sr
    import tempfile
    import scipy.io.wavfile
    recognizer = sr.Recognizer()
    duration = 10
    fs = 16000
    if st.button("Start Listening"):
        try:
            audio = record_audio(duration=duration, fs=fs)
            audio_np = audio.flatten()
            buf = plot_waveform(audio_np, fs)
            st.image(buf, caption='Audio Waveform', use_column_width=True)
            # Save to temp wav file for recognition
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                scipy.io.wavfile.write(tmpfile.name, fs, audio)
                with sr.AudioFile(tmpfile.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        st.success(f"You said: {text}")
                    except sr.UnknownValueError:
                        st.error("Sorry, I could not understand the audio.")
                    except sr.RequestError as e:
                        st.error(f"Could not request results; {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Microphone or audio error: {e}")

elif selected == "Image Recognition":
    st.header("Image Recognition")
    st.write("Upload an image and the app will try to recognize what is in the image.")
    uploaded_file = st.file_uploader("Choose an image for recognition...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        from PIL import Image
        import torch
        import torchvision.transforms as transforms
        import torchvision.models as models
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Recognizing...")
        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)
        # Load model
        model = models.resnet50(pretrained=True)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # Load labels
        import json
        import urllib.request
        LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(LABELS_URL) as f:
            categories = [line.strip() for line in f.readlines()]
        # Show top 3
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        for i in range(top3_prob.size(0)):
            st.write(f"{categories[top3_catid[i]].decode('utf-8')}: {top3_prob[i].item()*100:.2f}%")
