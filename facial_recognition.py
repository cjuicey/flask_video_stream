import cv2
import numpy as np

class FacialRecognition():
    def __init__(self):
        cascPath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def bytes_to_rgb(self, img_bin):
        # image in bytes format to rgb array
        img_arr = np.frombuffer(img_bin, dtype=np.uint8)
        img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB) # blues and greens may be swapped
        return img_arr, img_rgb

    def overlay_face_box(self, img_bin, faces):
        img_arr, img_rgb = bytes_to_rgb(img_bin)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img_rgb.tobytes()

    def detect_face(self, img_bin):
        # bin -> numpy -> bin
        img_arr, _ = bytes_to_rgb(img_bin)
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY) 
        faces = self.faceCascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        return faces
