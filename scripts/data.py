import glob
import os
from zipfile import ZipFile

import requests
import tensorflow as tf


def file_based_patch_generator(
    data_dir, crop_size=192, scale=4, n_patches_per_image=10
):
    """Generate patches directly from image files without loading whole dataset into memory"""
    # Get all image files
    files = glob.glob(os.path.join(data_dir, "*.png")) + glob.glob(
        os.path.join(data_dir, "*.jpg")
    )

    for file_path in files:
        # Load single image
        img = tf.io.read_file(file_path)
        hr_img = tf.image.decode_image(img, channels=3)
        hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)
        hr_shape = tf.shape(hr_img)

        # Only process images that are large enough
        if hr_shape[0] > crop_size and hr_shape[1] > crop_size:
            for _ in range(n_patches_per_image):
                # Random crop
                y = tf.random.uniform([], 0, hr_shape[0] - crop_size, dtype=tf.int32)
                x = tf.random.uniform([], 0, hr_shape[1] - crop_size, dtype=tf.int32)
                hr_patch = hr_img[y : y + crop_size, x : x + crop_size, :]

                # Create LR patch by downscaling
                lr_patch = tf.image.resize(
                    hr_patch, [crop_size // scale, crop_size // scale], method="area"
                )

                yield lr_patch, hr_patch


def get_file_patch_dataset(data_dir, crop_size=192, scale=4, n_patches_per_image=10):
    """Create dataset from file-based patch generator"""
    output_signature = (
        tf.TensorSpec(
            shape=(crop_size // scale, crop_size // scale, 3), dtype=tf.float32
        ),
        tf.TensorSpec(shape=(crop_size, crop_size, 3), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        lambda: file_based_patch_generator(
            data_dir, crop_size, scale, n_patches_per_image
        ),
        output_signature=output_signature,
    )


def download_data():
    urls = [
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    ]

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    for url in urls:
        filename = os.path.join(data_dir, os.path.basename(url))
        # Download zip file
        print(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download ready:", filename)

        # Exctract zip file
        print(f"Extracting {filename}...")
        with ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extracted files!")

    print("Data pulling ready:", data_dir)
