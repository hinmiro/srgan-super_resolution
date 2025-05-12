import keras
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
vgg.trainable = False
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)


def preprocess_vgg(x):
    x = tf.image.resize(x, [224, 224])
    x = preprocess_input(x * 255.0)
    return x


def combined_loss(y_true, y_pred):
    # Preprocess for VGG
    y_true_vgg = preprocess_vgg(y_true)
    y_pred_vgg = preprocess_vgg(y_pred)

    # Extract features
    f_true = vgg_model(y_true_vgg)
    f_pred = vgg_model(y_pred_vgg)

    # Perceptual loss: feature-wise MSE
    perceptual = tf.reduce_mean(tf.square(f_true - f_pred))

    # Normalize perceptual loss
    perceptual /= tf.cast(tf.size(f_true), tf.float32)

    pixel = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 20 * perceptual + 1.0 * pixel


@register_keras_serializable()
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


@register_keras_serializable()
def load_model(filepath):
    return keras.models.load_model(
        filepath,
        custom_objects={
            "PixelShuffle": PixelShuffle,
            "combined_loss": combined_loss,
            "PSNR": PSNR,
            "SSIM": SSIM,
            "PiecewiseConstantDecay": PiecewiseConstantDecay,
        },
    )


@register_keras_serializable()
def PSNR(original, generated):
    return tf.image.psnr(original, generated, max_val=1.0)


@register_keras_serializable()
def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)
