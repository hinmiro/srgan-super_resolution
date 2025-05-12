# SRGAN Super Resolution Model

This project implements an SRGAN-style super-resolution model using TensorFlow and Keras.  
The base code is adapted from a previous group project on EDSR Super Resolution.  
The [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is used for training and evaluation.

## Features

- SRGAN generator and discriminator architectures
- Data pipeline with random cropping, augmentation, batching, and prefetching
- Training with perceptual and adversarial loss
- Evaluation using PSNR and SSIM metrics
- Visualization of results

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/hinmiro/srgan-super_resolution
    cd super_resolution
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the dataset:**  
   The script will automatically download and prepare the DIV2K dataset using TensorFlow Datasets.

2. **Train the model:**
    ```sh
    python main.py
    ```

3. **Evaluate and visualize results:**  
   Evaluation scripts and plotting functions are included in the project.

## Project Structure

- `main.py` — Main training and evaluation script
- `scripts/data.py` — Data loading and preprocessing
- `models/` — Model architectures
- `utils/` — Helper functions, loss functions, evaluation, etc.
- `requirements.txt` — Python dependencies