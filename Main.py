import logging
from os import listdir
from os.path import isfile, join

import cv2

import ImageUtils
import ListUtils
from KeypointsDetector import KeypointsDetector
from KeypointsMatcher import KeypointsMatcher
from Orthophoto import Orthophoto
from Stitching import Stitching

BATCH_SIZE = 2
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
                     orthophoto2: Orthophoto, i) -> Orthophoto:
    matches = kp_matcher.match(orthophoto1, orthophoto2)
    matches = sorted(matches, key=lambda val: val.distance)[:300]

    # img_matches = ImageUtils.drawMatches(orthophoto1, orthophoto2, matches)
    # cv2.imwrite(f"result_images/img_matches{get_index()}.jpg", img_matches)

    stitching_img = Stitching(orthophoto1, orthophoto2, matches).stitch_orthophotos()

    new_orthophoto = Orthophoto(stitching_img, compress_ratio=1.)
    kp_detector.detect_and_compute(new_orthophoto)

    return new_orthophoto


def create_orthophoto(kp_matcher, kp_detector, orthophotos):
    if len(orthophotos) == 1:
        return orthophotos[0]

    orthophoto = None

    for i in range(1, len(orthophotos)):
        if i == 1:
            # первая склейка, берем первые два изображения
            orthophoto = match_and_stitch(kp_matcher, kp_detector, orthophotos[i - 1], orthophotos[i], i)
        else:
            # следующие склейки, берем ортофото и текущее изображение
            orthophoto = match_and_stitch(kp_matcher, kp_detector, orthophoto, orthophotos[i], i)

        index = get_index()
        logging.info(f"Create Orthophoto{index}")
        cv2.imwrite(f"result_images/orthophoto{index}.jpg", orthophoto.img)

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
            orthophoto = Orthophoto(cv2.imread(full_file_name), compress_ratio=0.4)
            kp_detector.detect_and_compute(orthophoto)
            orthophotos.append(orthophoto)

            logging.debug(f"Processed {file_name}")
            photos_count += 1

        if limit is not None and photos_count == limit:
            break

    new_orthophotos = []
    for i, orthophotos_batch in enumerate(ListUtils.batch(orthophotos, BATCH_SIZE)):
        logging.info(f'Process {i} batch')
        result = create_orthophoto(kp_matcher, kp_detector, orthophotos_batch)
        new_orthophotos.append(result)

    create_orthophoto(kp_matcher, kp_detector, new_orthophotos)


if __name__ == "__main__":
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/orthophoto")
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/merlischachen")
    # start("/Users/alexbelogurow/Study/4sem/geotagged-images", limit=10)
    # start("/Users/alexbelogurow/Study/4sem/photos_non_gcp", limit=10)
    # start("/Users/alexbelogurow/Study/4sem/nirs/resources/rotated")
    # start("/Users/alexbelogurow/Study/4sem/geotagged-images", limit=30)
    start("/Users/alexbelogurow/Study/4sem/geotagged-2")
