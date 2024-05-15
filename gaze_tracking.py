from __future__ import division
import os
import cv2
import dlib
import numpy as np
from eye import Eye
from calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.left_rgion = None
        self.right_region = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)


    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)
        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)
        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def left_idle_right(self):
        try:
            var = (self.eye_left.horizontal + self.eye_right.horizontal) / 2
            if var <= 0.4:
                return "Right"
            elif var >= 0.6:
                return "Left"
            return "Idle"
        except:
            return "Idle"

    def up_idle_down(self):
        try:
            var = (self.eye_left.vertical + self.eye_right.vertical) / 2
            if var <= 0.45:
                return "Down"
            elif var >= 0.58:
                return "Up"
            return "Idle"
        except:
            return "Idle"


    def is_blinking(self):
        try:
            return self.eye_left.blinking > 4.0 and self.eye_right.blinking > 4.0
        except:
            return False

