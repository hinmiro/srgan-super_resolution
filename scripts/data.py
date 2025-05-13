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


# Trying not to oom GPU
def patch_generator(split, crop_size=192, scale=4, n_patches_per_image=10):
    ds = tfds.load("div2k/bicubic_x4", split=split, as_supervised=True)
    for lr_img, hr_img in ds:
        hr_shape = tf.shape(hr_img)
        for _ in range(n_patches_per_image):
            # Random crop coordinates
            y = tf.random.uniform([], 0, hr_shape[0] - crop_size, dtype=tf.int32)
            x = tf.random.uniform([], 0, hr_shape[1] - crop_size, dtype=tf.int32)
            hr_patch = hr_img[y : y + crop_size, x : x + crop_size, :]
            lr_patch = tf.image.resize(
                hr_patch, [crop_size // scale, crop_size // scale], method="area"
            )
            yield lr_patch, hr_patch


def get_patch_dataset(split, crop_size=192, scale=4, n_patches_per_image=10):
    output_signature = (
        tf.TensorSpec(
            shape=(crop_size // scale, crop_size // scale, 3), dtype=tf.float32
        ),
        tf.TensorSpec(shape=(crop_size, crop_size, 3), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(
        lambda: patch_generator(split, crop_size, scale, n_patches_per_image),
        output_signature=output_signature,
    )


def get_test_dataset(dataset):
    val_count = 100
    val = dataset.take(val_count // 2)
    test = dataset.skip(val_count // 2)
    return test
