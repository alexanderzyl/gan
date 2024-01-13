import os

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt

# create the discriminator model
def create_discriminator(n_inputs=2):
    inp = Input(shape=(n_inputs,))
    x = Dense(25, activation='relu')(inp)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_generator(latent_dim, n_outputs=2):
    inp = Input(shape=(latent_dim,))
    x = Dense(15, activation='relu')(inp)
    x = Dense(n_outputs, activation='linear')(x)
    model = Model(inputs=inp, outputs=x)
    return model


def create_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Model(inputs=generator.input, outputs=discriminator(generator.output))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def real_function(x):
    return x * x


def generate_real_samples(n):
    X1 = np.random.rand(n) - 0.5
    X2 = real_function(X1)
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y


def generate_fake_samples(generator, latent_dim, n):
    x_input = np.random.randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    X = generator.predict(x_input)
    y = np.zeros((n, 1))
    return X, y


def generate_latent_points(latent_dim, n):
    return np.random.randn(latent_dim * n).reshape(n, latent_dim)


latent_shape = 5
n_size = 2
discriminator = create_discriminator(n_inputs=n_size)
generator = create_generator(latent_shape, n_outputs=n_size)
gan = create_gan(generator, discriminator)


def summarize_performance(epoch, g_model, d_model, latent_dim, n=100):
    x_real, y_real = generate_real_samples(n)
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print("Epoch: %d, Accuracy real: %.0f%%, Accuracy fake: %.0f%%" % (epoch, acc_real * 100, acc_fake * 100))
    # scatter plot real and fake data points
    plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    # create the folder plot if not exists
    if not os.path.exists('plot'):
        os.makedirs('plot')

    plt.savefig('plot/epoch_%03d.png' % (epoch + 1))
    plt.close()


def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)
        if (i + 1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim)


train(generator, discriminator, gan, latent_shape, n_eval=100)
