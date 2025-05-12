import numpy as np

from utils.util import PixelShuffle
from keras import layers, Model
from models.blocks import SRGAN_ResBlock, PixelShuffle


def build_generator(num_filters=64, num_res_blocks=16, scale=4):
    input_layer = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(num_filters, 9, padding="same")(input_layer)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    res = x

    # Add residual blocks
    for _ in range(num_res_blocks):
        res = SRGAN_ResBlock(res, num_filters)

    res = layers.Conv2D(num_filters, 3, padding="same")(res)
    res = layers.BatchNormalization()(res)
    x = layers.Add()([x, res])

    # Add Up sampling blocks
    for _ in range(int(np.log2(scale))):
        x = layers.Conv2D(num_filters * 4, 3, padding="same")(x)
        x = PixelShuffle(2)(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)

    output_layer = layers.Conv2D(3, 9, padding="same", activation="tanh")(x)
    return Model(inputs=input_layer, outputs=output_layer)


# SRGAN discriminator
def build_srgan_discriminator(input_shape=(None, None, 3), num_filters=64):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(num_filters, 3, strides=1, padding="same")(inp)
    x = layers.LeakyReLU(0.2)(x)

    filters = num_filters
    for i in range(1, 4):
        x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        filters *= 2

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out)
