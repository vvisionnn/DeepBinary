import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Deep_binary.model.unet_model import unet_very_little, unet_little
from Deep_binary.loss.loss import dice_coef, dice_2_coef


def processing_image(img_path):
    in_img = cv2.imread(img_path)
    # read image
    in_img = cv2.cvtColor(
        cv2.imread(img_path),
        cv2.COLOR_BGR2GRAY
    )
    _, in_img = cv2.threshold(in_img, 0, 255, cv2.THRESH_OTSU)
    origin_shape = in_img.shape

    # cut in image
    origin_img_height = in_img.shape[0]
    origin_img_weight = in_img.shape[1]

    if (origin_img_height % 256 != 0) or (origin_img_weight % 256 != 0):
        origin_img_height = (origin_img_height // 256 + 1) * 256 + 1
        origin_img_weight = (origin_img_weight // 256 + 1) * 256 + 1
        in_img = cv2.resize(in_img, (origin_img_weight, origin_img_height))

    pre_row, pre_line = 0, 0
    print(in_img.shape)
    print(origin_img_height)
    print(origin_img_weight)
    h = in_img.shape[1]
    w = in_img.shape[0]
    for row in range(256, w, 256):
        for line in range(256, h, 256):
            model_in = in_img[pre_row:row, pre_line:line] / 255.0
            model_in = np.expand_dims(model_in, axis=2)
            model_in = np.expand_dims(model_in, axis=0)
            model_in = tf.convert_to_tensor(model_in)
            print(model_in.shape)

            model_in = model.predict(model_in)
            in_img[pre_row:row, pre_line:line] = np.squeeze(model_in) * 255.0
            pre_line = line if h - line > 256 else 0
        pre_row = row if w - row > 256 else 0

    cv2.imwrite('img_processing.png', cv2.resize(in_img, (origin_shape[1], origin_shape[0])))


model = unet_little()
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef, dice_2_coef]
)
model.load_weights("/Users/zw/Downloads/deep_binary_ver0.9_best_loss.h5")
model.summary()
processing_image("/Users/zw/Downloads/dataset/4_in.png")
