import tensorflow as tf
import tensorflow_datasets as tfds


def get_div2k_dataset(scale=4, test_split=True, data_dir=None):
    kwargs = {"data_dir": data_dir} if data_dir else {}
    train = tfds.load(
        f"div2k/bicubic_x{scale}", split="train", as_supervised=True, **kwargs
    )
    val_full = tfds.load(
        f"div2k/bicubic_x{scale}", split="validation", as_supervised=True, **kwargs
    )

    val_count = 100
    if test_split:
        val = val_full.take(val_count // 2)
        test = val_full.skip(val_count // 2)
    else:
        val = val_full
        test = None

    return train, val, test
