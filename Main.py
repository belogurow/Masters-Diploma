import logging
import time
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

import ImageUtils
import ListUtils
from KeypointsDetector import KeypointsDetector
from KeypointsMatcher import KeypointsMatcher
from Orthophoto import Orthophoto
from Stitching import Stitching

BATCH_SIZE = 3
GLOBAL_INDEX = 0

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(filename)s:%(lineno)d \t%(levelname)s - %(message)s',
)


def get_index():
    global GLOBAL_INDEX
    GLOBAL_INDEX += 1
    return GLOBAL_INDEX


def match_and_stitch(kp_matcher: KeypointsMatcher, kp_detector: KeypointsDetector, orthophoto1: Orthophoto,
                     orthophoto2: Orthophoto) -> Orthophoto:
    matches = kp_matcher.match(orthophoto1, orthophoto2)
    matches = sorted(matches, key=lambda val: val.distance)[:300]

    # img_matches = ImageUtils.drawMatches(orthophoto1, orthophoto2, matches)
    # cv2.imwrite(f"result_images/img_matches{get_index()}.jpg", img_matches)

    stitching_img = Stitching(orthophoto1, orthophoto2, matches).stitch_orthophotos()

    new_orthophoto_name = f'{orthophoto1.img_name}+{orthophoto2.img_name}'
    new_orthophoto = Orthophoto(stitching_img, new_orthophoto_name, compress_ratio=1.)
    kp_detector.detect_and_compute(new_orthophoto)

    return new_orthophoto


def create_orthophoto(kp_matcher, kp_detector, orthophotos):
    if len(orthophotos) == 1:
        return orthophotos[0]

    orthophoto = None

    for i in range(1, len(orthophotos)):
        if i == 1:
            # первая склейка, берем первые два изображения
            orthophoto = match_and_stitch(kp_matcher, kp_detector, orthophotos[i - 1], orthophotos[i])
        else:
            # следующие склейки, берем ортофото и текущее изображение
            orthophoto = match_and_stitch(kp_matcher, kp_detector, orthophoto, orthophotos[i])

        index = get_index()
        logging.info(f"Create Orthophoto{index}")
        ImageUtils.save_img(f"orthophoto{index}", orthophoto.img)
        # cv2.imwrite(f"result_images/orthophoto{index}.jpg", orthophoto.img)

    return orthophoto


def start(image_path_folder, limit=None):
    if limit is not None:
        assert limit >= 2

    orthophotos = []

    kp_detector = KeypointsDetector()
    kp_matcher = KeypointsMatcher()

    # Для всех входных фото определяем их ключевые точки
    photos_count = 0
    for file_name in sorted(listdir(image_path_folder)):
        full_file_name = join(image_path_folder, file_name)

        if isfile(full_file_name) and ImageUtils.is_image(full_file_name):
            orthophoto = Orthophoto(cv2.imread(full_file_name), file_name, compress_ratio=0.4)
            kp_detector.detect_and_compute(orthophoto)
            orthophotos.append(orthophoto)

            logging.debug(f"Processed {file_name}")
            photos_count += 1

        if limit is not None and photos_count == limit:
            break

    new_orthophotos = []
    batch_idx = 0
    # while True:
    #     for orthophotos_batch in ListUtils.batch(orthophotos, BATCH_SIZE):
    #         logging.info(f'Process {batch_idx} batch')
    #         batch_idx += 1
    #
    #         result = create_orthophoto(kp_matcher, kp_detector, orthophotos_batch)
    #         new_orthophotos.append(result)
    #
    #     if len(new_orthophotos) == 1:
    #         break
    #
    #     # [:] copy without reference
    #     orthophotos = new_orthophotos[:]
    #     new_orthophotos.clear()

    for orthophotos_batch in ListUtils.batch(orthophotos, BATCH_SIZE):
        logging.info(f'Process {batch_idx} batch')
        batch_idx += 1

        result = create_orthophoto(kp_matcher, kp_detector, orthophotos_batch)
        new_orthophotos.append(result)

    create_orthophoto(kp_matcher, kp_detector, new_orthophotos)


def find_path(orthophoto_path, image_path_folder, limit=None):
    if limit is not None:
        assert limit >= 1

    kp_detector = KeypointsDetector()
    kp_matcher = KeypointsMatcher()

    proccessed_photos = []

    # Для всех входных фото определяем их ключевые точки
    photos_count = 0
    for file_name in sorted(listdir(image_path_folder)):
        full_file_name = join(image_path_folder, file_name)

        if isfile(full_file_name) and ImageUtils.is_image(full_file_name):
            orthophoto = Orthophoto(cv2.imread(full_file_name), file_name, compress_ratio=0.4)
            kp_detector.detect_and_compute(orthophoto)
            proccessed_photos.append(orthophoto)

            logging.debug(f"Processed {file_name}")
            photos_count += 1

        if limit is not None and photos_count == limit:
            break

    main_orthophoto = Orthophoto(cv2.imread(orthophoto_path), orthophoto_path, compress_ratio=1)
    kp_detector.detect_and_compute(main_orthophoto)

    centers = []

    for proccessed_photo in proccessed_photos:
        matches = kp_matcher.match(proccessed_photo, main_orthophoto)
        matches = sorted(matches, key=lambda val: val.distance)[:300]

        # img_matches = ImageUtils.drawMatches(main_orthophoto, proccessed_photo, matches)
        # ImageUtils.save_img("matches", img_matches)

        # ключевые точки на исходном изображении и их проекции на втором
        src_points = np.float32([proccessed_photo.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dest_points = np.float32([main_orthophoto.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # матрица проективной плоскости переводящие одни точки в другие
        M, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)

        # высота, длина исходного изображения
        h, w = proccessed_photo.height, proccessed_photo.width
        border_img1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # центр изображения
        center_img1 = np.float32([proccessed_photo.height / 2, proccessed_photo.width / 2]).reshape(-1, 1, 2)

        # рамка первого изображения на втором
        border_img1 = cv2.perspectiveTransform(border_img1, M)
        center_img1 = cv2.perspectiveTransform(center_img1, M).reshape(2)
        centers.append(center_img1)

        # result = cv2.polylines(main_orthophoto.img, [np.int32(border_img1)], True, 255, 10, cv2.LINE_AA)
        result = cv2.circle(main_orthophoto.img, (int(center_img1[0]), int(center_img1[1])), radius=20,
                            color=(0, 255, 255), thickness=10)
        # ImageUtils.save_img("path_image", result)

    for i in range(len(centers) - 1):
        center1 = (int(centers[i][0]), int(centers[i][1]))
        center2 = (int(centers[i + 1][0]), int(centers[i + 1][1]))
        cv2.line(main_orthophoto.img, center1, center2, color=(0, 255, 255), thickness=3)

    ImageUtils.save_img("result", main_orthophoto.img)


if __name__ == "__main__":
    start_time = time.time()
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/orthophoto")
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/merlischachen")
    # start("/Users/alexbelogurow/Study/4sem/geotagged-images", limit=10)
    # start("/Users/alexbelogurow/Study/4sem/drone-photos/photos_non_gcp")
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/rotated")
    # start("/Users/alexbelogurow/Study/4sem/drone-photos/geotagged-images", limit=100)
    # start("/Users/alexbelogurow/Study/4sem/drone-photos/geotagged-2")

    find_path(orthophoto_path="/Users/alexbelogurow/Study/4sem/drone-photos/non_gcp_orthophoto.jpg",
              image_path_folder="/Users/alexbelogurow/Study/4sem/drone-photos/photos_non_gcp_path")

    end_time = time.time()
    logging.info(f'Time of execution {end_time - start_time} sec')
