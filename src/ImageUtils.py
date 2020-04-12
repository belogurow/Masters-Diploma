import cv2
import numpy as np

from Orthophoto import Orthophoto


def crop_image_only_outside(img, tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]

def drawMatches(orthophoto1: Orthophoto, orthophoto2: Orthophoto, matches):
    img1 = orthophoto1.img
    img2 = orthophoto2.img
    kp1 = orthophoto1.keypoints
    kp2 = orthophoto2.keypoints

    rows1 = img1.shape[0]  # y1
    cols1 = img1.shape[1]  # x1
    rows2 = img2.shape[0]  # y2
    cols2 = img2.shape[1]  # x2

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1])
    out[:rows2, cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1), int(y1)), 4, (0, 0, 255, 1), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (0, 0, 255, 1), 1)
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 0, 255, 1), 2)

    return out


def is_image(image_path):
    return image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png")


def is_not_transparent_pixels(img):
    return img[..., [3]] == 255
