from Deep_binary.model.unet_model import unet
import cv2
import numpy as np
import tensorflow as tf

model = unet()

model.load_weights("/Users/zw/Downloads/version_0.6_weights.h5")
# model.summary()


def get_in_image(img_path):
    img = cv2.cvtColor(
        cv2.imread(img_path),
        cv2.COLOR_BGR2GRAY
    )

    img_out = np.zeros(shape=img.shape)

    # cut in image
    origin_img_height = img.shape[0]
    origin_img_weight = img.shape[1]
    pre_row, pre_line = 0, 0
    for row in range(256, origin_img_height, 256):
        for line in range(256, origin_img_weight, 256):
            model_in = img[pre_row:row, pre_line:line] / 255.0
            model_in = np.expand_dims(model_in, axis=2)
            model_in = np.expand_dims(model_in, axis=0)
            model_in = tf.convert_to_tensor(model_in)
            model_in = model.predict(model_in)
            img_out[pre_row:row, pre_line:line] = np.squeeze(model_in) * 255.0
            pre_line = line if origin_img_weight - line > 256 else 0
        pre_row = row if origin_img_height - row > 256 else 0

    cv2.namedWindow("Origin", cv2.WINDOW_NORMAL)
    cv2.imshow("Origin", img)
    cv2.namedWindow("After processing", cv2.WINDOW_NORMAL)
    cv2.imshow("After processing", img_out)
    cv2.waitKey()

    cv2.destroyAllWindows()


get_in_image("/Users/zw/Downloads/dataset/332_in.png")


