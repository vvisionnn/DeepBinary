# ======== version 0.8 ========
import os

import cv2
import numpy as np
import tensorflow as tf

# define super parameters
IMG_HEIGHT = 256
IMG_WEIGHT = 256
BATCH_SIZE = 16


def norm(img):
    return img / 255.0


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

        # yield little image
        origin_img_height = in_image.shape[0]
        origin_img_weight = in_image.shape[1]
        pre_row, pre_line = 0, 0
        for row in range(img_height, origin_img_height, img_height):
            for line in range(img_weight, origin_img_weight, img_weight):
                yield (np.expand_dims(norm(in_image[pre_row:row, pre_line:line]), axis=2),
                       np.expand_dims(norm(gt_image[pre_row:row, pre_line:line]), axis=2))
                # print(f"[{pre_row}, {row}]\t[{pre_line}, {line}]")
                pre_line = line if origin_img_weight - line > img_height else 0
            pre_row = row if origin_img_height - row > img_weight else 0


def load_dataset(ds_path):
    return tf.data.Dataset.from_generator(
        lambda: load_image_dataset(ds_path, IMG_HEIGHT, IMG_WEIGHT),
        (tf.float32, tf.float32),
        ((IMG_HEIGHT, IMG_WEIGHT, 1), (IMG_HEIGHT, IMG_WEIGHT, 1))
    ).shuffle(10000).repeat().batch(BATCH_SIZE)


# # ======== version 0.7 ========
# import os
#
# import tensorflow as tf
# import numpy as np
#
#
# # define super parameters
# IMG_HEIGHT = 512
# IMG_WEIGHT = 512
# IMG_CHANNEL = 1
# BATCH_SIZE = 16
# TRAIN_VAL_PERCENT = 0.7
#
#
# def get_all_image(all_img_path: str) -> (list, list):
#     all_img_num = int(len(os.listdir(all_img_path)) / 2)
#     img_x_list, img_y_list = [], []
#     for num in range(all_img_num):
#         img_x_list.append(str(os.path.join(all_img_path, str(num) + "_in.png")))
#         img_y_list.append(str(os.path.join(all_img_path, str(num) + "_gt.png")))
#
#     return img_x_list, img_y_list
#
#
# def shuffle_all_image(img_x: list, img_y: list) -> (list, list):
#     img_x = np.asarray(img_x)
#     img_y = np.asarray(img_y)
#
#     assert img_x.shape == img_y.shape
#
#     indices = np.arange(len(img_x))
#     np.random.shuffle(indices)
#     return img_x[indices].tolist(), img_y[indices].tolist()
#
#
# def split_train_validation(img_x: list, img_y: list) -> ((list, list), (list, list)):
#     assert len(img_x) == len(img_y)
#     train_num = int(len(img_x) * TRAIN_VAL_PERCENT)
#     return (img_x[:train_num], img_y[:train_num]), \
#            (img_x[train_num:], img_y[train_num:])
#
#
# def get_gray_image_tensor(img_path):
#     return tf.image.rgb_to_grayscale(
#         tf.image.decode_image(tf.io.read_file(img_path))
#     )
#
#
# # define image augmentation function
# def aug_image(img: tf.Tensor, seed: int) -> tf.Tensor:
#     img = tf.image.rot90(img, seed)
#     img = tf.image.flip_left_right(img, seed)
#     img = tf.image.flip_up_down(img, seed)
#     return img
#
#
# # define load dataset generator with infinite loop
# def load_generator(x_input, y_input):
#     while True:
#         count = 0
#         for in_img_path, gt_img_path in zip(x_input, y_input):
#             # read image and convert to gray scale
#             in_image = get_gray_image_tensor(in_img_path)
#             gt_image = get_gray_image_tensor(gt_img_path)
#
#             origin_img_height = in_image.shape[0]
#             origin_img_weight = in_image.shape[1]
#             pre_row, pre_line = 0, 0
#
#             for row in range(IMG_HEIGHT, origin_img_height, IMG_HEIGHT):
#                 for line in range(IMG_WEIGHT, origin_img_weight, IMG_WEIGHT):
#                     roi_in = in_image[pre_row:row, pre_line:line]
#                     roi_gt = gt_image[pre_row:row, pre_line:line]
#
#                     yield aug_image(roi_in, count), aug_image(roi_gt, count)
#
#                     pre_line = line if origin_img_weight - line > IMG_WEIGHT else 0
#                 pre_row = row if origin_img_height - row > IMG_HEIGHT else 0
#         count += 1
#         count %= 4
#
#
# # define load dataset function
# def load_dataset(x, y):
#     return tf.data.Dataset.from_generator(
#         lambda: load_generator(x, y),
#         (tf.float32, tf.float32),
#         ((IMG_HEIGHT, IMG_WEIGHT, IMG_CHANNEL),
#          (IMG_HEIGHT, IMG_WEIGHT, IMG_CHANNEL))
#     ).shuffle(10000).batch(BATCH_SIZE)
#
#
# if __name__ == '__main__':
#     # load all image
#     x_img, y_img = get_all_image("/Users/zw/Downloads/dataset")
#     # shuffle all image
#     x_img, y_img = shuffle_all_image(x_img, y_img)
#     # split train and validation dataset
#     (x_train, y_train), (x_test, y_test) = split_train_validation(x_img, y_img)
#     print(x_train)
#     print(y_train)
#     print(x_test)
#     print(y_test)
#     print(len(x_train))
#     print(len(y_train))
#     print(len(x_test))
#     print(len(y_test))
#


# def load_image_dataset(ds_path, img_height: int = 512, img_weight: int = 512):
#     img_num = int(len(os.listdir(ds_path)) / 2)
#     for num in range(img_num):
#         # read image and convert to gray scale
#         in_image = cv2.cvtColor(
#             cv2.imread(os.path.join(ds_path, str(num) + "_in.png")),
#             cv2.COLOR_BGR2GRAY
#         )
#         gt_image = cv2.cvtColor(
#             cv2.imread(os.path.join(ds_path, str(num) + "_gt.png")),
#             cv2.COLOR_BGR2GRAY
#         )
#         # ensure the shape of in_image equal to the shape of gt_image
#         assert in_image.shape == gt_image.shape
#         assert in_image.shape == gt_image.shape
#
#         # yield little image
#         origin_img_height = in_image.shape[0]
#         origin_img_weight = in_image.shape[1]
#         pre_row, pre_line = 0, 0
#         for row in range(img_height, origin_img_height, img_height):
#             for line in range(img_weight, origin_img_weight, img_weight):
#                 yield (np.expand_dims(in_image[pre_row:row, pre_line:line], axis=2),
#                        np.expand_dims(gt_image[pre_row:row, pre_line:line], axis=2))
#                 # print(f"[{pre_row}, {row}]\t[{pre_line}, {line}]")
#                 pre_line = line if origin_img_weight - line > img_height else 0
#             pre_row = row if origin_img_height - row > img_weight else 0


# def load_dataset(ds_path):
#     return tf.data.Dataset.from_generator(
#         lambda: load_image_dataset(ds_path, IMG_HEIGHT, IMG_WEIGHT),
#         (tf.float32, tf.float32),
#         ((512, 512, 1), (512, 512, 1))
#     ).shuffle(5000).repeat().batch(BATCH_SIZE)

