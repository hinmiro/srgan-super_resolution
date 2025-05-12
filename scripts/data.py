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
    val_list = list(val_full)
    val_size = len(val_list)
    split_idx = val_size // 2

    val = tf.data.Dataset.from_tensor_slices(val_list[:split_idx])
    test = (
        tf.data.Dataset.from_tensor_slices(val_list[split_idx:]) if test_split else None
    )

    return train, val, test
