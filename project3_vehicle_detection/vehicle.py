import cv2
import numpy as np

class Vehicle:
    """
    2D Vehicle defined by top-left and bottom-right corners.
    """
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_side = self.x_max - self.x_min
        self.y_side = self.y_max - self.y_min


    def draw(self, frame, color=255, thickness=1):
        """
        Draw Vehicle on a given frame.
        """
        cv2.rectangle(frame, (self.x_min, self.y_min), (self.x_max, self.y_max), color, thickness)