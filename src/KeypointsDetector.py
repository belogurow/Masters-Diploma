import cv2

from Orthophoto import Orthophoto


class KeypointsDetector:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=5000, scaleFactor=2, nlevels=8, edgeThreshold=20,
                                       scoreType=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    def detect_and_compute(self, orthophoto: Orthophoto):
        orthophoto.keypoints, orthophoto.descriptors = self.detector.detectAndCompute(orthophoto.gray_img, None)
