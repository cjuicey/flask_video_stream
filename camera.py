import picamera
import io
import numpy as np
import cv2
from time import time

from facial_recognition import FacialRecognition

class FakeNoiseCamera(object):
    """ Generates RGB static noise frames """
    def __init__(self):
        self.frame = self.gen_image()
        self.time = time()
        self.fps = 60
        self.frame_time_trigger = round(1/self.fps, 4)

    def get_frame(self):
        # updates every second
        if time() - self.time > self.frame_time_trigger:
            self.frame = self.gen_image()
            self.time = time()
        return self.frame

    def gen_image(self):
        imsize = (256, 256, 3) # 640x480 rgb image
        imarray = np.random.randint(0,255, imsize)#/255
        _, im_buf_arr = cv2.imencode(".jpg", imarray)
        byte_im = im_buf_arr.tobytes()
        return byte_im

class RealCamera(object):
    def __init__(self):
        self.camera, self.stream = self.initialize_video_stream()
        self.frame = self.get_frame()
        self.fps = 5
        self.frame_time_trigger = round(1/self.fps,4)

    def initialize_video_stream(self):
        camera = picamera.PiCamera()
        stream = picamera.PiCameraCircularIO(camera, seconds=4)
        
        # settings
        camera.vflip = True
        camera.resolution = (256,256)

        camera.start_recording(stream, format='h264')
        return camera, stream

    def get_frame(self):
        base_img = io.BytesIO()
        self.camera.capture(base_img, format='jpeg')
        base_img.seek(0)
        img_bin = base_img.read()
        return img_bin

class FacialDetectionCamera(object):
    def __init__(self):
        self.camera, self.stream = self.initialize_video_stream()
        self.fr_model = FacialRecognition()
        self.faces = None # detected faces
        self.frame = self.get_frame()
        self.fps = 5
        self.frame_time_trigger = round(1/self.fps,4)


    def initialize_video_stream(self):
        camera = picamera.PiCamera()
        stream = picamera.PiCameraCircularIO(camera, seconds=4)
        
        # settings
        camera.vflip = True
        camera.resolution = (256,256)

        camera.start_recording(stream, format='h264')
        return camera, stream

    def get_frame(self):
        base_img = io.BytesIO()
        self.camera.capture(base_img, format='jpeg')
        base_img.seek(0)
        img_bin = base_img.read()
        img_bin = self.facial_detection_process(img_bin)
        return img_bin

    def facial_detection_process(self, img_bin):
        # condition on when to process frame
        # passes_per_second = 1
        if time() - int(time()) < 0.1:
           self.faces = self.fr_model.detect_faces(img_bin)
        if self.faces:
            img_bin = self.fr_model.overlay_face_box(img_bin, faces)
        return img_bin 
