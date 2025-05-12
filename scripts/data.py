import tensorflow as tf
import tensorflow_datasets as tfds


def get_div2k_dataset(scale=4, data_dir=None):

    kwargs = {"data_dir": data_dir} if data_dir else {}
    train = tfds.load(
        f"div2k/bicubic_x{scale}", split="train", as_supervised=True, **kwargs
    )
    val = tfds.load(
        f"div2k/bicubic_x{scale}", split="validation", as_supervised=True, **kwargs
    )
    return train, val
