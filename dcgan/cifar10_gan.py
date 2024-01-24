from typing import Tuple

import numpy as np
from keras import Model, Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.src.initializers.initializers import RandomNormal, RandomUniform

from dcgan.generic import DcGan


class Cifar10Gan(DcGan):
    def create_discriminator(self) -> Model:
        in_shape = (32, 32, 3)
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def create_generator(self) -> Model:
        n_nodes = 256 * 4 * 4
        init = RandomNormal(stddev=0.02)
        # init = RandomUniform(minval=-0.05, maxval=0.05)
        model = Sequential()
        model.add(Dense(n_nodes, input_dim=self.latent_dim, kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))

        model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # output layer
        model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

        return model

    def generate_real_samples(self, dataset, n_samples):
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = np.ones((n_samples, 1))
        return X, y

    def generate_latent_points(self, latent_dim, n_samples):
        return np.random.randn(latent_dim * n_samples).reshape(n_samples, latent_dim)

    def generate_fake_samples(self, n_samples: int) -> Tuple[np.array, np.array]:
        latent_points = self.generate_latent_points(self.latent_dim, n_samples)
        X = self.generator.predict(latent_points)
        y = np.zeros((n_samples, 1))
        return X, y
