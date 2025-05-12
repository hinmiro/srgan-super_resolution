import tensorflow as tf
import tensorflow_datasets as tfds


def get_div2k_dataset(scale=4, cached=True, test_split=False):

    div2K_data = tfds.image.Div2k(config="bicubic_x{scale}")
    div2K_data.download_and_prepare()

    train = div2K_data.as_dataset(split="train", as_supervised=True)
    val = div2K_data.as_dataset(split="validation", as_supervised=True)

    if test_split:
        val = val.enumerate()
        test = val.filter(lambda i, _: i < 50).map(lambda i, x: x)
        val = val.filter(lambda i, _: i >= 50).map(lambda i, x: x)
    else:
        test = None

    if cached:
        train = train.cache()
        val = val.cache()
        if test:
            test = test.cache()

    return train, val, test
