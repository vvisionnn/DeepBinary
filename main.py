from Deep_binary.model.unet_model import unet
from Deep_binary.dataset.load_dataset import load_dataset
from Deep_binary.loss.loss import *
import tensorflow as tf


if __name__ == '__main__':
    train_ds, val_ds = load_dataset("/Users/zw/Downloads/dataset/")
    model = unet()
