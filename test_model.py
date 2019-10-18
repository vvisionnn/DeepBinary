import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):
    """Count Sorensen-Dice coefficient for output and ground-truth image."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    """Count loss of Sorensen-Dice coefficient for output and ground-truth image."""
    return 1 - dice_coef(y_true, y_pred)


def jacard_coef(y_true, y_pred):
    """Count Jaccard coefficient for output and ground-truth image."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    """Count loss of Jaccard coefficient for output and ground-truth image."""
    return 1 - jacard_coef(y_true, y_pred)


def load_model(model_path):
    model = keras.models.load_model(model_path)
    model.summary()
    return model


load_model("/Users/zw/Downloads/deep_binary_ver0.4.h5")

