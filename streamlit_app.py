import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from streamlit_option_menu import option_menu
from st_audiorec import st_audiorec
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

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

def record_audio_webrtc():
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
        def recv(self, frame):
            self.frames.append(frame.to_ndarray())
            return frame
    ctx = webrtc_streamer(key="audio", audio_receiver_size=1024, audio_processor_factory=AudioProcessor)
    if ctx.audio_receiver:
        st.info("Recording... Press Stop above to finish.")
        while ctx.state.playing:
            pass
        # Concatenate all frames
        audio_np = np.concatenate(ctx.audio_processor.frames, axis=0)
        return audio_np, ctx.audio_receiver.get_audio_frame_rate()
    return None, None

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
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_container_width=True)
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
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
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
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()

# --- Speech to Text (use streamlit_webrtc audio recorder) ---
elif selected == "Speech to Text":
    st.header("Speech to Text Recognition")
    st.write("Record your voice and transcribe it. The waveform of your speech will be shown.")
    import speech_recognition as sr
    import tempfile
    import scipy.io.wavfile
    recognizer = sr.Recognizer()
    audio_np, fs = record_audio_webrtc()
    if audio_np is not None:
        buf = plot_waveform(audio_np, fs)
        buf.seek(0)
        st.image(buf.read(), caption='Audio Waveform', use_container_width=True)
        # Save to temp wav file for recognition
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            scipy.io.wavfile.write(tmpfile.name, fs, audio_np)
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

elif selected == "Image Recognition":
    st.header("Image Recognition")
    st.write("Upload two images. The app will check if the second image is present in the first image using template matching.")
    uploaded_file1 = st.file_uploader("Choose the main image...", type=["jpg", "jpeg", "png"], key="main_img")
    uploaded_file2 = st.file_uploader("Choose the template image...", type=["jpg", "jpeg", "png"], key="template_img")
    if uploaded_file1 is not None and uploaded_file2 is not None:
        main_image = Image.open(uploaded_file1).convert('RGB')
        template_image = Image.open(uploaded_file2).convert('RGB')
        main_np = np.array(main_image)
        template_np = np.array(template_image)
        main_gray = cv2.cvtColor(main_np, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template_np, cv2.COLOR_RGB2GRAY)
        # Resize template if it's larger than main image
        if template_gray.shape[0] > main_gray.shape[0] or template_gray.shape[1] > main_gray.shape[1]:
            st.error("Template image is larger than the main image. Please upload a smaller template.")
        else:
            # Try multi-scale template matching for better robustness
            best_val = -1
            best_loc = None
            best_scale = 1.0
            h, w = template_gray.shape
            for scale in np.linspace(0.5, 1.5, 20):
                resized = cv2.resize(template_gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                if resized.shape[0] > main_gray.shape[0] or resized.shape[1] > main_gray.shape[1]:
                    continue
                res = cv2.matchTemplate(main_gray, resized, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale
                    best_shape = resized.shape
            threshold = 0.6
            found = best_val >= threshold
            result_img = main_np.copy()
            if found:
                h_scaled, w_scaled = best_shape
                top_left = best_loc
                bottom_right = (top_left[0] + w_scaled, top_left[1] + h_scaled)
                cv2.rectangle(result_img, top_left, bottom_right, (0,255,0), 3)
                st.success(f"Template found in main image! Match confidence: {best_val*100:.2f}% (scale={best_scale:.2f})")
            else:
                st.error(f"Template NOT found in main image. Best match confidence: {best_val*100:.2f}%")
            st.image([main_image, template_image], caption=["Main Image", "Template Image"], use_container_width=True)
            st.image(result_img, caption="Result (Green box = match)", use_container_width=True)
