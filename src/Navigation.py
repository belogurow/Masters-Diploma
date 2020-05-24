import logging
from os import listdir
from os.path import join, isfile

import cv2
import numpy as np

import ImageUtils
from KeypointsDetector import KeypointsDetector
from KeypointsMatcher import KeypointsMatcher
from Orthophoto import Orthophoto


class Navigation:
    def __init__(self, orthophoto_path, image_path_folder, start_point, end_point):
        self.kp_detector = KeypointsDetector()
        self.kp_matcher = KeypointsMatcher()

        self.orthophoto_path = orthophoto_path
        self.main_orthophoto = Orthophoto(cv2.imread(orthophoto_path), orthophoto_path, compress_ratio=1)

        self.image_path_folder = image_path_folder

        # todo возможно нужна проверка на координаты
        self.start_point = start_point
        self.end_point = end_point

    def start(self):
        proccessed_photos = []
        angles_between_points = [int(self.find_angle_between_points(self.start_point, self.end_point))]

        logging.info(
            f"Start navigation: distance {ImageUtils.distance_between_points(self.start_point, self.end_point)} px")
        logging.debug(f"First angle of direction {angles_between_points[-1]}")

        # Для всех входных фото определяем их ключевые точки
        photos_count = 0
        for file_name in sorted(listdir(self.image_path_folder)):
            full_file_name = join(self.image_path_folder, file_name)

            if isfile(full_file_name) and ImageUtils.is_image(full_file_name):
                orthophoto = Orthophoto(cv2.imread(full_file_name), file_name, compress_ratio=0.4)
                self.kp_detector.detect_and_compute(orthophoto)
                proccessed_photos.append(orthophoto)

                logging.debug(f"Processed {file_name}")
                photos_count += 1

        self.kp_detector.detect_and_compute(self.main_orthophoto)

        # рисуем начальную и конечную точки
        cv2.circle(self.main_orthophoto.img, self.start_point, radius=20,
                   color=(0, 255, 255), thickness=10)
        cv2.circle(self.main_orthophoto.img, self.end_point, radius=30,
                   color=(0, 255, 255), thickness=10)
        # линия между ними
        cv2.arrowedLine(self.main_orthophoto.img, self.start_point, self.end_point, color=(0, 255, 255), thickness=3,
                        tipLength=0.07)

        centers = []

        # определяем центры новых фото
        for proccessed_photo in proccessed_photos[:5]:
            matches = self.kp_matcher.match(proccessed_photo, self.main_orthophoto)
            matches = sorted(matches, key=lambda val: val.distance)[:300]

            # img_matches = ImageUtils.drawMatches(main_orthophoto, proccessed_photo, matches)
            # ImageUtils.save_img("matches", img_matches)

            # ключевые точки на исходном изображении и их проекции на втором
            src_points = np.float32([proccessed_photo.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dest_points = np.float32([self.main_orthophoto.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # матрица проективной плоскости переводящие одни точки в другие
            M, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)

            # высота, длина исходного изображения
            h, w = proccessed_photo.height, proccessed_photo.width
            border_img1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # центр изображения
            center_img1 = np.float32([proccessed_photo.height / 2, proccessed_photo.width / 2]).reshape(-1, 1, 2)

            # рамка первого изображения на втором
            # border_img1 = cv2.perspectiveTransform(border_img1, M)
            center_img1 = cv2.perspectiveTransform(center_img1, M).reshape(2)
            centers.append(center_img1)

            # Логирование местоположения дрона
            logging.info(
                f"Found current location on {proccessed_photo.img_name} - {center_img1}, distance {ImageUtils.distance_between_points(center_img1, self.end_point)} px")

            # Угол, на который надо повернуть дрону для корректировки полета (против часовой)
            angles_between_points.append(self.find_angle_between_points(center_img1, self.end_point))
            logging.debug(f"Angle of direction {angles_between_points[-1] - angles_between_points[-2]}")

            cv2.circle(self.main_orthophoto.img, (int(center_img1[0]), int(center_img1[1])), radius=15,
                       color=(0, 100, 255), thickness=10)
            cv2.line(self.main_orthophoto.img, (int(center_img1[0]), int(center_img1[1])), self.end_point,
                     color=(0, 255, 255),
                     thickness=3)
            # ImageUtils.save_img("path_image", result)

        centers.insert(0, self.start_point)
        centers.append(self.end_point)
        for i in range(len(centers) - 1):
            center1 = (int(centers[i][0]), int(centers[i][1]))
            center2 = (int(centers[i + 1][0]), int(centers[i + 1][1]))
            cv2.line(self.main_orthophoto.img, center1, center2, color=(255, 255, 255), thickness=3)

        logging.info("Stop navigation")

        return self.main_orthophoto

    def find_angle_between_points(self, start_point, end_point):
        # предполагается, что верктор скорости при начале навигации всегда направлен вверх
        # необходимо вычислить, на какой угол надо повернуть дрону, чтобы двигаться в направлении конечной точки
        # угол вычисляется по направлению от вертикального вектора прочитв часов стрелки

        vector1 = np.array(end_point) - np.array(start_point)

        # точка на самом вверху
        upper_point = (start_point[0], 0)
        vector2 = np.array(upper_point) - np.array(start_point)

        angle = np.math.atan2(np.linalg.det([vector1, vector2]), np.dot(vector1, vector2))
        return np.degrees(angle)
