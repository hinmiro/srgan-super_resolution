import os
import random
import shutil

import tensorflow as tf

from models.model import build_generator, build_srgan_discriminator
from scripts.data import get_file_patch_dataset, download_data
from scripts.training import stage1_train, stage_2_train
from utils.evaluation import evaluate, plot_comparison
from utils.helper_functions import (
    add_random_noise,
    flip_left_right,
    random_rotate,
)
from utils.loss_functions import set_dis_optimizer, set_gen_optimizer
from keras import Model


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    print("---SRGAN Super resolution model---")

    # Hyperparameters
    CROP_SIZE = 192
    SCALE = 4
    BATCH_SIZE = 2

    y_n = input("Download dataset? Y/N")
    if y_n.upper() == "Y":
        download_data()

    # Data locations
    train = "./data/DIV2K_train_HR/"
    validation = "./data/DIV2K_valid_HR/"
    test = "./data/DIV2K_test_HR"
    os.makedirs(test, exist_ok=True)

    for path in [train, validation, test]:
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist!")

    train_files = os.listdir(train)
    num_test_files = int(len(train_files) * 0.15)
    files_to_move = random.sample(train_files, num_test_files)

    if not os.listdir(test):
        for file in files_to_move:
            src = os.path.join(train, file)
            dst = os.path.join(test, file)
            shutil.move(src, dst)
        print(f"Moved {len(files_to_move)} files to test directory")
    else:
        print("Test files already exist")

    # Load datasets
    train_ds = get_file_patch_dataset(
        train, crop_size=CROP_SIZE, scale=SCALE, n_patches_per_image=4
    )
    val_ds = get_file_patch_dataset(
        validation, crop_size=CROP_SIZE, scale=SCALE, n_patches_per_image=2
    )
    test_ds = get_file_patch_dataset(
        test, crop_size=CROP_SIZE, scale=SCALE, n_patches_per_image=1
    )

    for lr, hr in train_ds.take(1):
        print("LR shape before crop:", lr.shape)
        print("HR shape before crop:", hr.shape)

    # Apply batching to data
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(1)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(1)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)

    # Print data shapes
    for lr, hr in train_ds.take(1):
        print("LR patch shape:", lr.shape)
        print("HR patch shape:", hr.shape)

    # Add noise to train data
    tf.print("Preparing dataset...")
    train_ds = train_ds.map(
        lambda lr, hr: (add_random_noise(lr), hr), num_parallel_calls=tf.data.AUTOTUNE
    )

    # Add data augmentation to train data
    train_ds = train_ds.map(
        lambda lr, hr: (random_rotate(lr, hr)), num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.map(
        lambda lr, hr: (flip_left_right(lr, hr)), num_parallel_calls=tf.data.AUTOTUNE
    )

    # Build models
    tf.print("Building models")
    discriminator = build_srgan_discriminator(
        input_shape=(CROP_SIZE, CROP_SIZE, 3), num_filters=64
    )
    generator = build_generator(num_filters=64, num_res_blocks=16, scale=4)

    # Set optimizers
    tf.print("Setting optimizers")
    set_dis_optimizer(learning_rate=5e-7)
    set_gen_optimizer(learning_rate=1e-4)

    # Stage 1 training with mae loss function
    load_weights = input("Load pretrained weights to skip pre training? Y/N: ")
    if load_weights.upper() == "N":
        tf.print("Starting phase 1 training...")
        stage1_train(generator, train_ds, val_ds, epochs=120)
    elif load_weights.upper() == "Y":
        generator.load_weights("./checkpoints/mae_pretrained.weights.h5")
    else:
        print("Invalid choice...")

    # Stage 2 training
    tf.print("Starting phase 2 training...")
    history = stage_2_train(
        generator, discriminator, train_ds, val_ds, patience=30, epoch=200
    )

    # Evaluate with test data
    evaluate(generator, test_ds)

    # Create comparison image
    test_samples = list(test_ds)
    lr, hr = random.choice(test_samples)
    sr = generator(lr, training=False)
    plot_comparison(lr, sr, hr)


if __name__ == "__main__":
    main()
