# ======== version 0.6 ========
import os

import tensorflow as tf
import cv2
import numpy as np


# define super parameters
IMG_HEIGHT = 512
IMG_WEIGHT = 512
BATCH_SIZE = 16


def load_image_dataset(ds_path, img_height: int = 512, img_weight: int = 512):
    img_num = int(len(os.listdir(ds_path)) / 2)
    for num in range(img_num):
        # read image and convert to gray scale
        in_image = cv2.cvtColor(
            cv2.imread(os.path.join(ds_path, str(num) + "_in.png")),
            cv2.COLOR_BGR2GRAY
        )
        gt_image = cv2.cvtColor(
            cv2.imread(os.path.join(ds_path, str(num) + "_gt.png")),
            cv2.COLOR_BGR2GRAY
        )
        # ensure the shape of in_image equal to the shape of gt_image
        assert in_image.shape == gt_image.shape
        assert in_image.shape == gt_image.shape

        # yield little image
        origin_img_height = in_image.shape[0]
        origin_img_weight = in_image.shape[1]
        pre_row, pre_line = 0, 0
        for row in range(img_height, origin_img_height, img_height):
            for line in range(img_weight, origin_img_weight, img_weight):
                yield (np.expand_dims(in_image[pre_row:row, pre_line:line], axis=2),
                       np.expand_dims(gt_image[pre_row:row, pre_line:line], axis=2))
                # print(f"[{pre_row}, {row}]\t[{pre_line}, {line}]")
                pre_line = line if origin_img_weight - line > img_height else 0
            pre_row = row if origin_img_height - row > img_weight else 0


def load_dataset(ds_path):
    return tf.data.Dataset.from_generator(
        lambda: load_image_dataset(ds_path, IMG_HEIGHT, IMG_WEIGHT),
        (tf.float32, tf.float32),
        ((512, 512, 1), (512, 512, 1))
    ).shuffle(5000).repeat().batch(BATCH_SIZE)



