import cv2
from camera import Camera
from face_detector import FaceDetector

def main():
    # Initialize the camera
    cam = Camera()
    face_detector = FaceDetector()

    # Start the camera feed
    cam.start()

    while True:
        # Capture frame-by-frame
        frame = cam.read_frame()
        if frame is None:
            break

        # Detect faces in the frame
        faces = face_detector.detect_faces(frame)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()