import cv2
import tensorflow as tf
import numpy as np

from loss.loss import dice_coef, dice_2_coef
from model.unet_model import unet_little, unet

# define super parameters
IMG_HEIGHT = 256
IMG_WEIGHT = 256
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


def read_tensor(img_path):
    img = cv2.cvtColor(
        cv2.imread(img_path),
        cv2.COLOR_BGR2GRAY
    )
    # binary image by OTSU
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # print(img.shape)
    # print(img)
    return img / 255.0

def predict_image(img_path):
    in_image = read_tensor(img_path)
    # get image height and width
    origin_img_height = in_image.shape[0]
    origin_img_width = in_image.shape[1]

    height = (origin_img_height // 255 + 1) * 255 + 1
    width = (origin_img_width // 255 + 1) * 255 + 1
    out_image = np.zeros((height, width))
    out_image[:origin_img_height, :origin_img_width] = in_image
    in_image = out_image
    in_image = np.expand_dims(in_image, axis=2)
    out_image_2 = np.zeros((height, width))
    pre_row, pre_line = 0, 0
    for row in range(IMG_HEIGHT, height, IMG_HEIGHT):
        for line in range(IMG_WEIGHT, width, IMG_WEIGHT):
            # print(f"[{pre_row}:{row}, {pre_line}:{line}]")
            in_img = tf.cast(in_image[pre_row:row, pre_line:line], tf.float32)
            in_img = tf.expand_dims(in_img, axis=0)
            pre_img = model.predict(in_img)
            pre_img = tf.squeeze(pre_img)
            out_image_2[pre_row:row, pre_line:line] =  pre_img * 255.0
            pre_line = line if origin_img_width - line > IMG_WEIGHT else 0
        pre_row = row if origin_img_height - row > IMG_HEIGHT else 0
    return np.asarray(out_image_2[:origin_img_height, :origin_img_width], np.uint8)


model = unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef, dice_2_coef]
)

model.load_weights("save_models/deep_binary_ver0.95.h5")
# model.load_weights("save_models/deep_binary_ver0.9_best_dice.h5")
# model.load_weights("save_models/deep_binary_ver0.9_best_dice_2.h5")

img = predict_image("dataset/all/145_in.png")
print(img.shape)
cv2.imwrite('145.png', img)

