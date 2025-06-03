import cv2
class Camera:
    def __init__(self):
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(0)

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read_frame(self):
        if self.capture is not None:
            ret, frame = self.capture.read()
            if ret:
                return frame
        return None