import tensorflow as tf
from keras.optimizers import Adam

from utils.util import preprocess_vgg, vgg_model

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = None
discriminator_optimizer = None


def set_gen_optimizer(learning_rate=1e-4):
    global generator_optimizer
    return Adam(learning_rate=learning_rate)


def set_dis_optimizer(learning_rate=5e-7):
    global discriminator_optimizer
    return Adam(learning_rate=learning_rate)


def discriminator_loss(real_output, fake_output):
    real_labels = tf.random.uniform(tf.shape(real_output), 0.8, 1.0)
    fake_labels = tf.random.uniform(tf.shape(fake_output), 0.0, 0.2)
    real_loss = bce(real_labels, real_output)
    fake_loss = bce(fake_labels, fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output, sr, hr):
    adv_loss = bce(tf.ones_like(fake_output), fake_output)

    # Perceptual loss
    vgg_hr = vgg_model(preprocess_vgg(hr))
    vgg_sr = vgg_model(preprocess_vgg(sr))
    perceptual_loss = tf.reduce_mean(tf.square(vgg_hr - vgg_sr))

    pixel_loss = tf.reduce_mean(tf.abs(hr - sr))

    return 2e-5 * adv_loss + 0.006 * perceptual_loss + pixel_loss


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
