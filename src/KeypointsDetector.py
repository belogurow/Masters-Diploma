import cv2

from Orthophoto import Orthophoto


class KeypointsDetector:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=5000, scaleFactor=2, nlevels=5, edgeThreshold=20,
                                       scoreType=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    def detect_and_compute(self, orthophoto: Orthophoto):
        orthophoto.keypoints, orthophoto.descriptors = self.detector.detectAndCompute(orthophoto.gray_img, None)

        # img_with_keypoints = cv2.drawKeypoints(orthophoto.img, orthophoto.keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # ImageUtils.save_img(f'{orthophoto.img_name}_keypoints', img_with_keypoints)
