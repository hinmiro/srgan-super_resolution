import os
import random
import shutil
from resource.banner import banner

import tensorflow as tf

from models.model import build_generator, build_srgan_discriminator
from scripts.data import construct_datasets, download_data, load_dataset
from scripts.training import stage1_train, stage_2_train
from utils.evaluation import evaluate, plot_comparison
from utils.loss_functions import set_dis_optimizer, set_gen_optimizer
from utils.util import load_model


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    print(banner)

    # Hyperparameters
    CROP_SIZE = 192
    SCALE = 4
    BATCH_SIZE = 2

    y_n = input("Download dataset? Y/N: ")
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
    train_ds, val_ds, test_ds = load_dataset("./data")
    train_sr, val_sr, test_sr = construct_datasets(
        train_ds, val_ds, test_ds, BATCH_SIZE, CROP_SIZE, SCALE
    )

    # Print data shapes
    for lr, hr in train_sr.take(1):
        print("LR patch shape:", lr.shape)
        print("HR patch shape:", hr.shape)

    # Build models
    tf.print("Building models")
    discriminator = build_srgan_discriminator(
        input_shape=(CROP_SIZE, CROP_SIZE, 3), num_filters=64
    )
    generator = build_generator(num_filters=64, num_res_blocks=16, scale=4)

    # Set optimizers
    tf.print("Setting optimizers")
    d_optimizer = set_dis_optimizer(learning_rate=5e-7)
    g_optimizer = set_gen_optimizer(learning_rate=1e-4)

    skip_train = input("Skip training and load pretrained model? Y/N: ")
    if skip_train.upper() == "N":

        # Stage 1 training with mae loss function
        load_weights = input("Load pretrained weights to skip pre training? Y/N: ")
        if load_weights.upper() == "N":
            tf.print("Starting phase 1 training...")
            stage1_train(generator, train_sr, val_sr, epochs=120)
        elif load_weights.upper() == "Y":
            generator.load_weights("./checkpoints/mae_pretrained.weights.h5")
        else:
            print("Invalid choice...")

        # Stage 2 training
        tf.print("Starting phase 2 training...")
        history = stage_2_train(
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            train_sr,
            val_sr,
            patience=30,
            epoch=200,
        )
    elif skip_train.upper() == "Y":
        generator = load_model("./checkpoints/srgan_generator.keras")

    else:
        print("Invalid choice...")

    # Evaluate with test data
    evaluate(generator, test_sr)

    # Create comparison image
    for lr, hr in test_sr.take(1):
        sr = generator(lr, training=False)
        plot_comparison(lr, sr, hr)


if __name__ == "__main__":
    main()
