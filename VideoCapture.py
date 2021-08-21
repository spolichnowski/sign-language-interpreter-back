from cv2 import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import expand_dims
import numpy as np
import operator


class VideoCapture:
    '''
    Class that creates video object that is later used
    in a Flask application.
    '''

    def __init__(self, resolution):
        ''' Initiate video capturing '''
        self.video = cv2.VideoCapture(0)
        self.height = int(resolution.split('x')[0])
        self.width = int(resolution.split('x')[1])
        #self.model = int(resolution.split('x')[2])

    def __del__(self):
        ''' Release video '''
        self.video.release()

    def get_video_capture(self):
        '''  
        Gets frame and its shape. Resizes the frame proportionally (here by 3)
        and flips the camera so it works like a mirror. Gets the prediction outcome
        and puts it with openCV on the screen as text. 
        In the end returns encoded video to than display it in Flask application. 
        '''

        ret, frame = self.video.read()
        h, w, l = frame.shape
       
        frame = cv2.resize(frame, (self.width, self.height))
        flipped_frame = cv2.flip(frame, 1)


        ret, jpeg = cv2.imencode('.jpg', flipped_frame)
        return flipped_frame, jpeg.tobytes()
