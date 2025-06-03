"""
Hand Gesture Detection Project
This script detects the number of fingers shown on both hands using a webcam and prints the total count.
"""
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Helper function to count fingers
# Returns the number of fingers up for a single hand
# hand_landmarks: mediapipe hand landmarks
# hand_label: 'Left' or 'Right'
def count_fingers(hand_landmarks, hand_label):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pip = [3, 6, 10, 14, 18]
    count = 0
    # Thumb
    if hand_label == 'Right':
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pip[0]].x:
            count += 1
    else:
        if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pip[0]].x:
            count += 1
    # Other fingers
    for tip, pip in zip(finger_tips[1:], finger_pip[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1
    return count

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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
            cv2.imshow('Hand Gesture Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
