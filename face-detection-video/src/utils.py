import cv2
def draw_rectangles(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

def resize_frame(frame, width):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    height = int(width / aspect_ratio)
    resized_frame = cv2.resize(frame, (width, height))
    return resized_frame