import glob
import os
import requests
import tensorflow as tf

from utils.helper_functions import load_image
from zipfile import ZipFile


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


def load_dataset(data_dir):
    train_dir = os.path.join(data_dir, "DIV2K_train_HR")
    val_dir = os.path.join(data_dir, "DIV2K_valid_HR")
    test_dir = os.path.join(data_dir, "DIV2K_test_HR")

    train_files = [
        os.path.join(train_dir, fname)
        for fname in os.listdir(train_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    val_files = [
        os.path.join(val_dir, fname)
        for fname in os.listdir(val_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    test_files = [
        os.path.join(test_dir, fname)
        for fname in os.listdir(test_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    train_ds = tf.data.Dataset.from_tensor_slices(train_files).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = tf.data.Dataset.from_tensor_slices(val_files).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )


def construct_datasets(train_files, val_files, test_files):
    train_ds = tf.data.Dataset.from_tensor_slices(train_files).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = tf.data.Dataset.from_tensor_slices(val_files).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
