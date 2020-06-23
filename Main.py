import logging
import time
import random
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

import cv2
import numpy as np

import ImageUtils
import ListUtils
from KeypointsDetector import KeypointsDetector
from KeypointsMatcher import KeypointsMatcher
from Navigation import Navigation
from NavigationConfig import NavigationConfig
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

    # Для всех входных фото определяем ключевые точки и их дескрипторы
    photos_count = 0
    for file_name in sorted(listdir(image_path_folder)):
        full_file_name = join(image_path_folder, file_name)

        if isfile(full_file_name) and ImageUtils.is_image(full_file_name):
            orthophoto = Orthophoto(cv2.imread(full_file_name), file_name, compress_ratio=0.1)
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


def start_navigation(orthophoto_path, config_file, image_path_folder):
    config = NavigationConfig(config_file)

    navigation = Navigation(orthophoto_path, image_path_folder, config)
    result = navigation.start()

    ImageUtils.save_img("result", result.img)


if __name__ == "__main__":
    start_time = time.time()
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/orthophoto")
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/merlischachen")
    # start("/Users/alexbelogurow/Study/4sem/geotagged-images", limit=10)
    # start("/Users/alexbelogurow/Study/4sem/drone-photos/photos_non_gcp")
    # start("/Users/alexbelogurow/Study/4sem/drone-photos/presa")
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/rotated")
    # start("/Users/alexbelogurow/Study/4sem/drone-photos/geotagged-images", limit=100)
    # start("/Users/alexbelogurow/Study/4sem/drone-photos/geotagged-2")

    # find_path(orthophoto_path="/Users/alexbelogurow/Study/4sem/drone-photos/non_gcp_orthophoto.jpg",
    #           image_path_folder="/Users/alexbelogurow/Study/4sem/drone-photos/photos_non_gcp_path")

    # start_navigation(orthophoto_path="/Users/alexbelogurow/Study/4sem/drone-photos/non_gcp_orthophoto.jpg",
    #                  image_path_folder="/Users/alexbelogurow/Study/4sem/drone-photos/photos_non_gcp_path",
    #                  start_point=(2500, 2000),
    #                  end_point=(500, 500))

    # start_navigation(
    #     orthophoto_path="/Users/alexbelogurow/Study/4sem/nirs/resources/navigation/non_gcp_orthophoto.jpg",
    #     config_file="/Users/alexbelogurow/Study/4sem/nirs/resources/navigation/configuration.json",
    #     image_path_folder="/Users/alexbelogurow/Study/4sem/drone-photos/photos_non_gcp_path")

    end_time = time.time()
    logging.info(f'Time of execution {end_time - start_time} sec')
