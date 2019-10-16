# import os
# from pathlib import Path
#
# import tensorflow as tf
#
#
# # define processing image function
# def process_image(image_raw, image_size):
#     image = tf.image.rgb_to_grayscale(tf.image.decode_png(image_raw))
#     image = tf.image.resize(image, image_size)
#     image /= 255.0
#     return image
#
#
# # load and process
# def load_and_process(img_path):
#     return process_image(tf.io.read_file(img_path), [256, 256])
#
#
# def load_dataset(ds_path: str):
#     images = []
#     labels = []
#     for num in range(int(len(os.listdir(ds_path)) / 2)):
#         images.append(num)
#         labels.append(num)
#
#     sorted(images)
#     sorted(labels)
#
#     images = list(map(lambda x: "dataset/" + str(x) + "_in.png", images))
#     labels = list(map(lambda x: "dataset/" + str(x) + "_gt.png", labels))
#
#     print(images)
#     print(labels)
#     AUTOTUNE = tf.data.experimental.AUTOTUNE
#
#     return tf.data.Dataset.zip((
#         tf.data.Dataset.from_tensor_slices(images).map(load_and_process, num_parallel_calls=AUTOTUNE),
#         tf.data.Dataset.from_tensor_slices(labels).map(load_and_process, num_parallel_calls=AUTOTUNE)
#     ))
#
#
# if __name__ == '__main__':
#     load_dataset("/Users/zw/Downloads/dataset/")
#
#
#


# ==================== version 0.2 ========================
import os

import numpy as np
import cv2
import tensorflow as tf


def norm(img):
    return img / 255.0


def process_one_image(img_path, img_height=256, img_weight=256):
    # read image
    image = cv2.imread(img_path)
    # convert image from bgr to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize image
    image = cv2.resize(image, (img_height, img_weight))
    # normalization image and return
    return np.expand_dims(norm(image), axis=3)


# define load all image function
def load_all_image(ds_path, img_height=256, img_weight=256):
    all_in_image = list(filter(lambda x: x.find('in') != -1, os.listdir(ds_path)))
    all_in_image = list(map(lambda x: "dataset/" + x, all_in_image))
    all_gt_image = list(map(lambda x: x.replace('in', 'gt'), all_in_image))

    all_in_image = np.asarray(all_in_image)
    all_gt_image = np.asarray(all_gt_image)

    assert len(all_in_image) == len(all_gt_image)

    # shuffle dataset
    indices = np.arange(len(all_in_image))
    np.random.shuffle(indices)
    all_in_image = all_in_image[indices]
    all_gt_image = all_gt_image[indices]
    print("length of all in image: ", len(all_in_image))
    print("length of all gt image: ", len(all_gt_image))

    return all_in_image, all_gt_image


# define load dataset function
def load_dataset(ds_path, img_height=256, img_weight=256):
    # load all image
    img_in, img_gt = load_all_image(ds_path, img_height, img_weight)
    # split train and test dataset
    assert len(img_in) == len(img_gt)

    img_in = list(map(process_one_image, img_in))
    img_gt = list(map(process_one_image, img_gt))

    train_num = int(len(img_in) * 0.8)
    x_train = tf.data.Dataset.from_tensor_slices(img_in[:train_num])
    y_train = tf.data.Dataset.from_tensor_slices(img_gt[:train_num])
    x_test = tf.data.Dataset.from_tensor_slices(img_in[train_num:])
    y_test = tf.data.Dataset.from_tensor_slices(img_gt[train_num:])

    train_ds = tf.data.Dataset.zip((x_train, y_train))
    test_ds = tf.data.Dataset.zip((x_test, y_test))
    return train_ds, test_ds


if __name__ == '__main__':
    train_ds, test_ds = load_dataset("/Users/zw/Downloads/dataset/")












