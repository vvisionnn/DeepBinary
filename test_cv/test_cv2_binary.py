import shutil

import cv2
import numpy as np


def get_roi(frame):
    roi = frame[:int(frame.shape[0] * 0.2),
                :int(frame.shape[1] * 0.2)]
    return roi


def load_image(img_path):
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(image)
    cv2.namedWindow("Hello", cv2.WINDOW_NORMAL)
    cv2.imshow("Hello", image)
    cv2.namedWindow("Hello_gt", cv2.WINDOW_NORMAL)
    gt_image = cv2.imread(img_path.replace("in", "gt"))
    cv2.namedWindow("Hello_gt", cv2.WINDOW_NORMAL)
    cv2.imshow("Hello_gt", gt_image)
    cv2.imshow("Hello_gt_roi", get_roi(gt_image))
    height = 256
    weight = 256
    channel = 3
    black_roi = np.zeros((height, weight, channel))
    cv2.imshow("black_roi", black_roi)
    gt_image[:height, :weight] = black_roi
    cv2.namedWindow("insert_black_roi", cv2.WINDOW_NORMAL)
    cv2.imshow("insert_black_roi", gt_image)

    cv2.waitKey()


# src = "/Users/zw/Downloads/dataset/"
# dst = "/Users/zw/Documents/py36/Deep_binary/test_cv/test_images/"
#
# for num in range(123, 133):
#     file_name = str(num) + "_gt.png"
#     print(file_name)
#     shutil.copy(src + file_name, dst + file_name)


# test opencv binary
load_image("test_images/124_in.png")

cv2.destroyAllWindows()

