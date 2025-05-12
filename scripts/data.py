import tensorflow as tf
import tensorflow_datasets as tfds


def get_div2k_dataset(scale=4, test_split=True, data_dir=None):

    kwargs = {"data_dir": data_dir} if data_dir else {}
    train = tfds.load(
        f"div2k/bicubic_x{scale}", split="train", as_supervised=True, **kwargs
    )
    val = tfds.load(
        f"div2k/bicubic_x{scale}", split="validation", as_supervised=True, **kwargs
    )
    test = (
        tfds.load(f"div2k/bicubic_x{scale}", split="test", as_supervised=True, **kwargs)
        if test_split
        else None
    )
    return train, val, test
