# # ==================== version 0.2 ========================
# import os
#
# import numpy as np
# import cv2
# import tensorflow as tf
#
#
# def norm(img):
#     return img / 255.0
#
#
# def process_one_image(img_path, img_height=256, img_weight=256):
#     # read image
#     image = cv2.imread(img_path)
#     # convert image from bgr to gray
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # resize image
#     image = cv2.resize(image, (img_height, img_weight))
#     # normalization image and return
#     return np.expand_dims(norm(image), axis=3)
#
#
# # define load all image function
# def load_all_image(ds_path, img_height=256, img_weight=256):
#     all_in_image = list(filter(lambda x: x.find('in') != -1, os.listdir(ds_path)))
#     all_in_image = list(map(lambda x: "dataset/" + x, all_in_image))
#     all_gt_image = list(map(lambda x: x.replace('in', 'gt'), all_in_image))
#
#     all_in_image = np.asarray(all_in_image)
#     all_gt_image = np.asarray(all_gt_image)
#
#     assert len(all_in_image) == len(all_gt_image)
#
#     # shuffle dataset
#     indices = np.arange(len(all_in_image))
#     np.random.shuffle(indices)
#     all_in_image = all_in_image[indices]
#     all_gt_image = all_gt_image[indices]
#     print("length of all in image: ", len(all_in_image))
#     print("length of all gt image: ", len(all_gt_image))
#
#     return all_in_image, all_gt_image
#
#
# # define load dataset function
# def load_dataset(ds_path, img_height=256, img_weight=256):
#     # load all image
#     img_in, img_gt = load_all_image(ds_path, img_height, img_weight)
#     # split train and test dataset
#     assert len(img_in) == len(img_gt)
#
#     img_in = list(map(process_one_image, img_in))
#     img_gt = list(map(process_one_image, img_gt))
#
#     train_num = int(len(img_in) * 0.8)
#     x_train = tf.data.Dataset.from_tensor_slices(img_in[:train_num])
#     y_train = tf.data.Dataset.from_tensor_slices(img_gt[:train_num])
#     x_test = tf.data.Dataset.from_tensor_slices(img_in[train_num:])
#     y_test = tf.data.Dataset.from_tensor_slices(img_gt[train_num:])
#
#     train_ds = tf.data.Dataset.zip((x_train, y_train))
#     test_ds = tf.data.Dataset.zip((x_test, y_test))
#     return train_ds, test_ds
#
#
# if __name__ == '__main__':
#     train_ds, test_ds = load_dataset("/Users/zw/Downloads/dataset/")
#


# ======== version 0.5 ========
import os
from pathlib import Path

import tensorflow as tf


def cut_image(raw_img, img_height: int=256, img_weight: int=256):
    img_stack = []
    origin_img_height = raw_img.shape[0]
    origin_img_weight = raw_img.shape[1]
    pre_row, pre_line = 0, 0
    for row in range(img_height, origin_img_height, img_height):
        for line in range(img_weight, origin_img_weight, img_weight):
            img_stack.append(raw_img[pre_row:row, pre_line:line])
            print(f"[{pre_row}, {row}]\t[{pre_line}, {line}]")
            print(raw_img[pre_row:row, pre_line:line])
            pre_line = line if origin_img_weight - line > img_height else 0
        pre_row = row if origin_img_height - row > img_weight else 0
    return img_stack


def load_and_cut_one_image(img_path):
    raw_img = tf.io.decode_png(tf.io.read_file(img_path))
    return cut_image(raw_img, 256, 256)


def load_all_image(ds_path: str) -> (list, list):
    ds_path = Path(ds_path)
    image_num = int(len(os.listdir(ds_path)) / 2)

    all_in_image, all_gt_image = [], []
    for i in range(image_num):
        in_image_path = ds_path / str(i) + "_in.png"
        gt_image_path = ds_path / str(i) + "_gt.png"
        all_in_image.extend(load_and_cut_one_image(in_image_path))
        all_gt_image.extend(load_and_cut_one_image(gt_image_path))
    return all_in_image, all_gt_image


all_in, all_gt = load_all_image("/Users/zw/Downloads/dataset")
print(all_in)
print(all_gt)
