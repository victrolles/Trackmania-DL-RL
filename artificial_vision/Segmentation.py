import numpy as np
import cv2
import os

class Segmentation:
    def __init__(self):
        self.b_color_interval = [110, 200]
        self.g_color_interval = [110, 190]
        self.r_color_interval = [110, 190]

        self.minim_pixels = 200000

    def segment_image(self, image):
        # Définition des bornes inférieures et supérieures en BGR
        lower_bound = np.array([self.b_color_interval[0], self.g_color_interval[0], self.r_color_interval[0]])
        upper_bound = np.array([self.b_color_interval[1], self.g_color_interval[1], self.r_color_interval[1]])

        mask = cv2.inRange(image, lower_bound, upper_bound)

        masked_image = cv2.bitwise_and(image, image, mask=mask)

        return mask, masked_image

