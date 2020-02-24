import cv2

from Orthophoto import Orthophoto


class KeypointsMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, orthophoto1: Orthophoto, orthophoto2: Orthophoto):
        return self.matcher.match(orthophoto1.descriptors, orthophoto2.descriptors)
