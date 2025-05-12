import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.util import PSNR, SSIM


def evaluate(generator, test_sr):
    psnr_vals, ssim_vals = [], []
    for lr, hr in test_sr:
        sr = generator(lr, training=False)
        psnr_vals.append(tf.reduce_mean(PSNR(hr, sr)).numpy())
        ssim_vals.append(tf.reduce_mean(SSIM(hr, sr)).numpy())
    print(f"Test PSNR: {np.mean(psnr_vals):.4f}")
    print(f"Test SSIM: {np.mean(ssim_vals):.4f}")


def plot_comparison(lr, sr, hr, filename="../data/comparison.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(12, 4))
    titles = ["Low-Res", "Super-Res", "High-Res"]
    images = [lr, sr, hr]
    for i, img in enumerate(images):
        plt.subplot(1, 3, i + 1)
        plt.imshow(tf.clip_by_value(img, 0, 1))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
