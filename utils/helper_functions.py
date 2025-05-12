import tensorflow as tf
import numpy as np

from utils.util import PSNR, SSIM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def flip_left_right(lowres_img, highres_img):
    """Flips Images to left and right."""

    # Outputs random values from a uniform distribution in between 0 to 1
    rn = tf.random.uniform(shape=(), maxval=1.0)

    # If rn is less than 0.5 it returns original lowres_img and highres_img
    # If rn is greater than 0.5 it returns flipped image
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )


def random_rotate(lowres_img, highres_img):
    """Rotates Images by 90 degrees."""

    # Outputs random values from uniform distribution in between 0 to 4
    rn = tf.cast(
        tf.random.uniform(shape=(), maxval=4, dtype=tf.float32), dtype=tf.int32
    )

    # Here rn signifies number of times the image(s) are rotated by 90 degrees
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    return tf.image.convert_image_dtype(img, tf.float32)


def random_crop_and_downscale(hr_img, crop_size, scale):
    hr_patch = tf.image.random_crop(hr_img, size=[crop_size, crop_size, 3])
    hr_patch.set_shape([crop_size, crop_size, 3])

    # Randomly apply blur
    if tf.random.uniform(()) > 0.5:
        hr_patch = tf.nn.avg_pool(hr_patch[None], ksize=3, strides=1, padding="SAME")[0]
    # Add random noise
    noise = tf.random.normal(tf.shape(hr_patch), mean=0.0, stddev=0.01)
    hr_patch = tf.clip_by_value(hr_patch + noise, 0.0, 1.0)

    lr_patch = tf.image.resize(
        hr_patch, [crop_size // scale, crop_size // scale], method="area"
    )
    lr_patch.set_shape([crop_size // scale, crop_size // scale, 3])
    return lr_patch, hr_patch


def add_random_noise_and(img):
    if tf.random.uniform(()) > 0.5:
        img = tf.nn.avg_pool(img[None], ksize=3, strides=1, padding="SAME")[0]

    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.01)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img


def random_crop_pair(lr, hr, hr_crop_size=92, scale=4):
    lr_crop_size = hr_crop_size // scale

    lr_shape = tf.shape(lr)
    lr_x = tf.random.uniform((), 0, lr_shape[0] - lr_crop_size + 1, dtype=tf.int32)
    lr_y = tf.random.uniform((), 0, lr_shape[1] - lr_crop_size + 1, dtype=tf.int32)

    # Crop lr image
    lr_cropped = lr[lr_x : lr_x + lr_crop_size, lr_y : lr_y + lr_crop_size, :]

    hr_x = lr_x * scale
    hr_y = lr_y * scale

    # Crop hr image
    hr_cropped = hr[hr_x : hr_x + hr_crop_size, hr_y : hr_y + hr_crop_size, :]

    return lr_cropped, hr_cropped


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    mode="min",
)
