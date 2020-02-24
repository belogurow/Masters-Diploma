import cv2
import numpy as np

import ImageUtils
from Orthophoto import Orthophoto


class Stitching:
    def __init__(self, orthophoto1: Orthophoto, orthophoto2: Orthophoto, matches):
        self.orthophoto1 = orthophoto1
        self.orthophoto2 = orthophoto2
        self.matches = matches

    def stitch_orthophotos(self):
        # ключевые точки на исходном изображении и их проекции на втором
        src_points = np.float32([self.orthophoto1.keypoints[m.queryIdx].pt for m in self.matches[:25]]).reshape(-1, 1,
                                                                                                                2)
        dest_points = np.float32([self.orthophoto2.keypoints[m.trainIdx].pt for m in self.matches[:25]]).reshape(-1, 1,
                                                                                                                 2)

        # матрица проективной плоскости переводящие одни точки в другие
        M, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)

        # высота, длина исходного изображения
        h, w = self.orthophoto1.height, self.orthophoto1.width
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # рамка первого изображения на втором
        dst = cv2.perspectiveTransform(pts, M)

        img1 = self.orthophoto1.img
        img2 = cv2.polylines(self.orthophoto2.img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # cv2.imwrite("original_image_overlapping.jpg", img2)

        # первое изображение в перспективе
        dst = cv2.warpPerspective(img1, M, (img2.shape[1] + img1.shape[1], img1.shape[0] + img2.shape[0]))
        # cv2.imwrite("persepctve.jpg", dst)

        # второе изображение накладываем на первое в перспективе
        dst[0:img2.shape[0], 0:img2.shape[1]] = img2
        # cv2.imwrite("stitching_img.jpg", dst)

        # обрезаем черные области по краям
        dst = ImageUtils.crop_image_only_outside(dst)

        return dst
