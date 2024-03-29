# model version 0.8
import tensorflow as tf
from tensorflow import keras


# define super parameters
IMG_HEIGHT = 256
IMG_WEIGHT = 256
BATCH_SIZE = 16


# model version 0.6
def double_conv(inputs, filters: int):
    conv_1 = keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    conv_relu_1 = keras.layers.Activation('relu')(conv_1)
    conv = keras.layers.BatchNormalization(axis=3)(conv_relu_1)
    conv_2 = keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(conv)
    conv_relu_2 = keras.layers.Activation('relu')(conv_2)
    conv_2 = keras.layers.BatchNormalization(axis=3)(conv_relu_2)
    return keras.layers.SpatialDropout2D(0.1)(conv_2)


# define down block
def unet_down_block(inputs, filters: int):
    conv = double_conv(inputs, filters)
    max_pool = keras.layers.MaxPooling2D((2, 2))(conv)
    return conv, max_pool


# define up block
def unet_up_block(inputs, concats, filters: int):
    return double_conv(
        tf.concat(
            [keras.layers.UpSampling2D((2, 2))(inputs), concats],
            axis=3
        ),
        filters
    )


# build unet architecture
def unet():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WEIGHT, 1))

    # left
    conv_1, maxpool_1 = unet_down_block(inputs, 64)
    conv_2, maxpool_2 = unet_down_block(maxpool_1, 128)
    conv_3, maxpool_3 = unet_down_block(maxpool_2, 256)
    conv_4, maxpool_4 = unet_down_block(maxpool_3, 512)

    # center
    center = double_conv(maxpool_4, 1024)

    # right
    up_sample_4 = unet_up_block(center, conv_4, 512)
    up_sample_3 = unet_up_block(up_sample_4, conv_3, 256)
    up_sample_2 = unet_up_block(up_sample_3, conv_2, 128)
    up_sample_1 = unet_up_block(up_sample_2, conv_1, 64)

    # output
    outputs = keras.layers.Conv2D(1, (1, 1))(up_sample_1)
    outputs = keras.layers.Activation('sigmoid')(outputs)

    return keras.models.Model(inputs, outputs)


def unet_little():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WEIGHT, 1))

    # left
    conv_1, maxpool_1 = unet_down_block(inputs, 64)
    conv_2, maxpool_2 = unet_down_block(maxpool_1, 128)
    conv_3, maxpool_3 = unet_down_block(maxpool_2, 256)

    # center
    center = double_conv(maxpool_3, 512)

    # right
    up_sample_3 = unet_up_block(center, conv_3, 256)
    up_sample_2 = unet_up_block(up_sample_3, conv_2, 128)
    up_sample_1 = unet_up_block(up_sample_2, conv_1, 64)

    # output
    outputs = keras.layers.Conv2D(1, (1, 1))(up_sample_1)
    outputs = keras.layers.Activation('sigmoid')(outputs)

    return keras.models.Model(inputs, outputs)