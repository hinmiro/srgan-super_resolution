import os
import time

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from rich.progress import Progress

from utils.helper_functions import early_stop, reduce_lr
from utils.loss_functions import (
    discriminator_loss,
    discriminator_optimizer,
    generator_loss,
    generator_optimizer,
)
from utils.util import PSNR, SSIM


def stage1_train(generator, train_data, val_data, epochs=100, loss="mae"):
    generator.compile(
        optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=[PSNR, SSIM]
    )
    generator.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
    )
    os.makedirs("./checkpoints", exist_ok=True)
    generator.save_weights("./checkpoints/mae_pretrained.weights.h5")


def validate(generator, val_sr):
    psnr_vals, ssim_vals = [], []
    for lr, hr in val_sr:
        sr = generator(lr, training=False)
        psnr_vals.append(tf.reduce_mean(PSNR(hr, sr)).numpy())
        ssim_vals.append(tf.reduce_mean(SSIM(hr, sr)).numpy())
    return np.mean(psnr_vals), np.mean(ssim_vals)


@tf.function
def train_step(lr, hr, generator, discriminator):
    with tf.GradientTape(persistent=True) as tape:
        sr = generator(lr, training=True)
        real_output = discriminator(hr, training=True)
        fake_output = discriminator(sr, training=True)

        # tf.print("real_output mean:", tf.reduce_mean(real_output))
        # tf.print("fake_output mean:", tf.reduce_mean(fake_output))
        # tf.print("real_output sample:", tf.reshape(real_output, [-1])[:10])
        # tf.print("fake_output sample:", tf.reshape(fake_output, [-1])[:10])

        d_loss = discriminator_loss(real_output, fake_output)
        g_loss = generator_loss(fake_output, sr, hr)

        grads_g = tape.gradient(g_loss, generator.trainable_variables)
        grads_d = tape.gradient(d_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(grads_d, discriminator.trainable_variables)
        )

        return g_loss, d_loss


def stage_2_train(generator, discriminator, train_sr, val_sr, patience=20, epoch=200):
    best_ssim = -np.inf
    wait = 0
    EPOCHS = epoch
    g_losses_history, d_losses_history = [], []

    with Progress(transient=False) as progress:
        for epoch in range(EPOCHS):
            start_time = time.time()
            g_losses, d_losses = [], []
            num_batches = len(list(train_sr))

            task = progress.add_task(f"[cyan]Epoch {epoch+1}", total=num_batches)
            for batch_idx, (lr, hr) in enumerate(train_sr):
                g_loss, d_loss = train_step(lr, hr, generator, discriminator)
                g_losses.append(g_loss.numpy())
                d_losses.append(d_loss.numpy())
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch+1}/{EPOCHS} | Batch: {batch_idx+1}/{num_batches} d_loss {d_loss:.3f}]",
                )

            # Validation
            val_psnr, val_ssim = validate(generator, val_sr)
            elapsed = int(time.time() - start_time)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            desc = (
                f"[cyan]Epoch {epoch+1}"
                f"[green] g_loss {np.mean(g_losses):.3f} d_loss {np.mean(d_losses):.3f}  PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.3f}[/green]"
                f"[cyan]/{elapsed_str}[/cyan]"
            )
            progress.update(task, description=desc)

            g_losses_history.append(np.mean(g_losses))
            d_losses_history.append(np.mean(d_losses))

            # Reduce learning rate
            if wait > 0 and wait % 5 == 0:
                old_lr = generator_optimizer.learning_rate.numpy()
                new_lr = max(old_lr * 0.5, 1e-6)
                generator_optimizer.learning_rate.assign(new_lr)
                discriminator_optimizer.learning_rate.assign(new_lr)
                progress.console.print(f"[red]Reduced learning rate to {new_lr}[/red]")

            # Early stop with ssim
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                wait = 0
                generator.save("srgan_generator_best.keras")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop activated at epoch {epoch+1}")
                    break

        generator.save("./checkpoints/srgan_generator.keras")
        return (g_losses_history, d_losses_history)
