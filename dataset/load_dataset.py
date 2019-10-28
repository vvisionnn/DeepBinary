import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import cv2
import numpy as np

# define super parameters
IMG_HEIGHT = 256
IMG_WEIGHT = 256
BATCH_SIZE = 8
TRAIN_TEST_SPLIT = 0.7


def get_train_test_indices(ds_path: str) -> (list, list):
    dir_num = len(os.listdir(ds_path))
    image_num = int(dir_num / 2)
    # shuffle indices to shuffle dataset
    indices = np.arange(image_num)
    np.random.shuffle(indices)
    # print(indices)
    train_num = int(image_num * TRAIN_TEST_SPLIT)
    # print(f"train num is {train_num}")
    return indices[:train_num], indices[train_num:]


def read_tensor(img_path):
    img = cv2.cvtColor(
        cv2.imread(img_path),
        cv2.COLOR_BGR2GRAY
    )
    # print(f"img.shape={img.shape}")
    # binary image by OTSU
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img = tf.expand_dims(img, axis=2)
    # cast opencv type to tf.tensor
    img = tf.cast(img, tf.float32)
    # print(img.shape)
    # print(img)
    return img / 255.0


def image_aug(img, rotate: int = 0):
    return tf.image.rot90(img, k=rotate)


def load_dataset_from_indices(ds_path: str, indices: list, is_train: bool = False):
    count = 0
    while True:
        for num in indices:
            # read gray image
            in_image = read_tensor(os.path.join(ds_path, str(num) + "_in.png"))
            gt_image = read_tensor(os.path.join(ds_path, str(num) + "_gt.png"))

            # get image height and width
            origin_img_height = in_image.shape[0]
            origin_img_width = in_image.shape[1]
            pre_row, pre_line = 0, 0
            for row in range(IMG_HEIGHT, origin_img_height, IMG_HEIGHT):
                for line in range(IMG_WEIGHT, origin_img_width, IMG_WEIGHT):
                    # print(f"[{pre_row}:{row}, {pre_line}:{line}]")
                    if is_train:
                        # image rotate
                        yield (
                            image_aug(in_image[pre_row:row, pre_line:line], count),
                            image_aug(gt_image[pre_row:row, pre_line:line], count)
                        )
                    else:
                        yield (
                            in_image[pre_row:row, pre_line:line],
                            gt_image[pre_row:row, pre_line:line]
                        )
                    pre_line = line if origin_img_width - line > IMG_WEIGHT else 0
                pre_row = row if origin_img_height - row > IMG_HEIGHT else 0
        count = count + 1 if count < 3 else 0


def load_dataset(ds_path: list):
    train_indices, test_indices = get_train_test_indices(ds_path)
    train_ds = tf.data.Dataset.from_generator(
        lambda: load_dataset_from_indices(ds_path, train_indices, is_train=True),
        (tf.float32, tf.float32),
        ((IMG_HEIGHT, IMG_WEIGHT, 1), (IMG_HEIGHT, IMG_WEIGHT, 1))
    ).shuffle(5000).repeat().batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_generator(
        lambda: load_dataset_from_indices(ds_path, test_indices),
        (tf.float32, tf.float32),
        ((IMG_HEIGHT, IMG_WEIGHT, 1), (IMG_HEIGHT, IMG_WEIGHT, 1))
    ).batch(BATCH_SIZE)
    return train_ds, test_ds


if __name__ == "__main__":
    count = 99
    img = None
    train_ds, test_ds = load_dataset("/home/py36/workspace/deep_binary/dataset/all")
    for img in test_ds:
        # count += 1
        count -= 1
        if count == 0: break

    img_1 = tf.squeeze(img[0])
    img_2 = tf.squeeze(img[1])
    print(f"img1.shape={img_1.shape}")
    print(f"img2.shape={img_2.shape}")

    cv2.imwrite('123.png', tf.squeeze(img_1[1]).numpy() * 255.0)
    cv2.imwrite('124.png', tf.squeeze(img_2[1]).numpy() * 255.0)

