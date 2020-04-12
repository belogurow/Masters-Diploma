import logging

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
        try:
            # ключевые точки на исходном изображении и их проекции на втором
            src_points = np.float32([self.orthophoto1.keypoints[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            dest_points = np.float32([self.orthophoto2.keypoints[m.trainIdx].pt for m in self.matches]).reshape(-1, 1,
                                                                                                                2)

            # матрица проективной плоскости переводящие одни точки в другие
            M, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)

            # высота, длина исходного изображения
            h, w = self.orthophoto1.height, self.orthophoto1.width
            border_img1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # рамка первого изображения на втором
            stitching_photo = cv2.perspectiveTransform(border_img1, M)

            # проверка на необходимость смещения изображения
            offset_x, offset_y, M = Stitching.test_offset(M, stitching_photo)

            img1 = self.orthophoto1.img
            img2 = self.orthophoto2.img
            # img2 = cv2.polylines(self.orthophoto2.img, [np.int32(border_img1)], True, 255, 10, cv2.LINE_AA)
            # cv2.imwrite("original_image_overlapping.jpg", img2)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

            # первое изображение в перспективе
            shape_for_perspective_image = (img1.shape[0] * 2 + img2.shape[0], img1.shape[1] * 2 + img2.shape[1])
            stitching_photo = cv2.warpPerspective(img1, M, dsize=shape_for_perspective_image)
            # cv2.imwrite("perspective.jpg", dst)

            # второе изображение накладываем на первое в перспективе
            # если пиксель второго изображения прозрачный - берем пиксель первого изображения
            replacing_area = stitching_photo[offset_y:offset_y + img2.shape[0], offset_x:offset_x + img2.shape[1]]
            replacing_area = np.where(ImageUtils.is_not_transparent_pixels(img2), img2, replacing_area)
            stitching_photo[offset_y:offset_y + img2.shape[0], offset_x:offset_x + img2.shape[1]] = replacing_area

            # обрезаем черные области по краям
            stitching_photo = ImageUtils.crop_image_only_outside(stitching_photo)

            return stitching_photo
        except Exception as e:
            # Если возникла ошибка, отдам первое изображение
            logging.error(e)
            return self.orthophoto1.img

    @staticmethod
    def test_offset(M, border_img1_on_img2):
        # Если границы вышли первого изображения вышли за пределы, то его необходимо подвинуть
        # то, есть модифировать homography matrix М

        min_x, min_y = min(border_img1_on_img2[:, 0, 0]), min(border_img1_on_img2[:, 0, 1])
        transformation_matrix = np.float32([[1, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 1]])

        if min_x < 0:
            offset_x = abs(int(min_x))
            transformation_matrix[0, 2] = offset_x
        else:
            offset_x = 0

        if min_y < 0:
            offset_y = abs(int(min_y))
            transformation_matrix[1, 2] = offset_y
        else:
            offset_y = 0

        M = transformation_matrix.dot(M)
        return offset_x, offset_y, M