# import tensorflow as tf
# from tensorflow import keras
#
#
# # model version 0.1
# # todo : update layers
# def double_conv(inputs, filters: int):
#     conv_1 = keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
#     conv_relu_1 = keras.layers.Activation('relu')(conv_1)
#     conv_2 = keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(conv_relu_1)
#     conv_relu_2 = keras.layers.Activation('relu')(conv_2)
#     return conv_relu_2
#
#
# # define down block
# def unet_down_block(inputs, filters: int):
#     conv = double_conv(inputs, filters)
#     max_pool = keras.layers.MaxPooling2D((2, 2))(conv)
#     return conv, max_pool
#
#
# # define up block
# def unet_up_block(inputs, concats, filters: int):
#     return double_conv(
#         tf.concat(
#             [keras.layers.UpSampling2D((2, 2))(inputs), concats],
#             axis=3
#         ),
#         filters
#     )
#
#
# # build unet architecture
# def unet():
#     inputs = keras.Input(shape=(256, 256, 1))
#
#     # left
#     conv_1, maxpool_1 = unet_down_block(inputs, 64)
#     conv_2, maxpool_2 = unet_down_block(maxpool_1, 128)
#     conv_3, maxpool_3 = unet_down_block(maxpool_2, 256)
#     conv_4, maxpool_4 = unet_down_block(maxpool_3, 512)
#
#     # center
#     center = double_conv(maxpool_4, 1024)
#
#     # right
#     up_sample_4 = unet_up_block(center, conv_4, 512)
#     up_sample_3 = unet_up_block(up_sample_4, conv_3, 256)
#     up_sample_2 = unet_up_block(up_sample_3, conv_2, 128)
#     up_sample_1 = unet_up_block(up_sample_2, conv_1, 64)
#
#     # output
#     outputs = keras.layers.Conv2D(1, (1, 1))(up_sample_1)
#     outputs = keras.layers.Activation('sigmoid')(outputs)
#
#     return keras.models.Model(inputs, outputs)
#
#
#
#

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import SpatialDropout2D, Activation, BatchNormalization
from tensorflow.keras.models import Model


def double_conv_layer(inputs, filter):
    conv = Conv2D(filter, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filter, (3, 3), padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = SpatialDropout2D(0.1)(conv)
    return conv


def down_layer(inputs, filter):
    """Create downsampling layer."""
    conv = double_conv_layer(inputs, filter)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def up_layer(inputs, concats, filter):
    """Create upsampling layer."""
    return double_conv_layer(tf.concat([UpSampling2D(size=(2, 2))(inputs), concats], axis=3), filter)


def unet():
    """Create U-net."""
    inputs = Input((256, 256, 1))

    # DownSampling.
    down1, pool1 = down_layer(inputs, 32)
    down2, pool2 = down_layer(pool1, 64)
    down3, pool3 = down_layer(pool2, 128)
    down4, pool4 = down_layer(pool3, 256)
    down5, pool5 = down_layer(pool4, 512)

    # Bottleneck.
    bottleneck = double_conv_layer(pool5, 1024)

    # UpSampling.
    up5 = up_layer(bottleneck, down5, 512)
    up4 = up_layer(up5, down4, 256)
    up3 = up_layer(up4, down3, 128)
    up2 = up_layer(up3, down2, 64)
    up1 = up_layer(up2, down1, 32)

    outputs = Conv2D(1, (1, 1))(up1)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs, outputs)

    return model


model = unet()
model.summary()

