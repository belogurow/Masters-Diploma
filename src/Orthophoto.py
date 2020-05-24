import cv2

import ImageUtils

# процент сжатия исходного изображения
DEFAULT_COMPRESS_RATIO = .5


class Orthophoto:
    def __init__(self, img, img_name=None, compress_ratio=DEFAULT_COMPRESS_RATIO):
        # self.img = img
        self.img = cv2.resize(img, (0, 0), fx=compress_ratio, fy=compress_ratio)
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.height, self.width, _ = self.img.shape
        self.img_name = ImageUtils.get_name_of_image(img_name) if img_name is not None else ''

        self.keypoints = None
        self.descriptors = None
