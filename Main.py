import logging
from os import listdir
from os.path import isfile, join

import cv2

import ImageUtils
from KeypointsDetector import KeypointsDetector
from KeypointsMatcher import KeypointsMatcher
from Orthophoto import Orthophoto
from Stitching import Stitching

# IMAGES_PATH = "resources/orthophoto/"
# IMAGES_NAME = ["DJI_0082.JPG",
#                "DJI_0084.JPG",
#                "DJI_0088.JPG"]


IMAGES_PATH = "resources/merlischachen/"
IMAGES_NAME = ["IMG_1183.JPG",
               "IMG_1185.JPG",
               "IMG_1186.JPG"]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(filename)s:%(lineno)d \t%(levelname)s - %(message)s',
)


def match_and_stitch(kp_matcher: KeypointsMatcher, kp_detector: KeypointsDetector, orthophoto1: Orthophoto,
                     orthophoto2: Orthophoto) -> Orthophoto:
    matches = kp_matcher.match(orthophoto1, orthophoto2)
    matches = sorted(matches, key=lambda val: val.distance)

    # img_matches = ImageUtils.drawMatches(orthophotos[i - 1], orthophotos[i], matches[:25])
    # cv2.imwrite(f"img_matches{i}.jpg", img_matches)

    stitching_img = Stitching(orthophoto1, orthophoto2, matches).stitch_orthophotos()

    new_orthophoto = Orthophoto(stitching_img, compress_ratio=1.)
    kp_detector.detect_and_compute(new_orthophoto)

    return new_orthophoto


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
            orthophoto = Orthophoto(cv2.imread(full_file_name), compress_ratio=0.4)
            kp_detector.detect_and_compute(orthophoto)
            orthophotos.append(orthophoto)

            logging.debug(f"Processed {file_name}")
            photos_count += 1

        if limit is not None and photos_count == limit:
            break

    # Если не было задано кол-во обрабатываемых фото, то берем общее их кол-во в папке
    if limit is None:
        limit = photos_count

    orthophoto = None
    for i in range(1, limit):
        if i == 1:
            # первая склейка, берем первые два изображения
            orthophoto = match_and_stitch(kp_matcher, kp_detector, orthophotos[i - 1], orthophotos[i])
        else:
            # следующие склейки, берем ортофото и текущее изображение
            orthophoto = match_and_stitch(kp_matcher, kp_detector, orthophoto, orthophotos[i])

        logging.info(f"Create Orthophoto{i}")
        cv2.imwrite(f"orthophoto{i}.jpg", orthophoto.img)


if __name__ == "__main__":
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/orthophoto")
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/merlischachen", limit=1)
    # start("/Users/alexbelogurow/Study/4sem/geotagged-images", limit=10)
    # start("/Users/alexbelogurow/Study/4sem/photos_non_gcp", limit=10)
    start("/Users/alexbelogurow/Study/4sem/nirs/resources/rotated", limit=2)
