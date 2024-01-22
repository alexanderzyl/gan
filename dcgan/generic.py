from typing import Tuple

import numpy as np
from keras import Model

from abc import ABC, abstractmethod


class DcGan(ABC):
    def __init__(self, latent_dim: int = 100):
        self.latent_dim = latent_dim
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()
        self.gan = self.create_gan()
        self.on_epoch_end = []

    @abstractmethod
    def create_discriminator(self) -> Model:
        pass

    @abstractmethod
    def create_generator(self) -> Model:
        pass

    @abstractmethod
    def create_gan(self) -> Model:
        pass

    @abstractmethod
    def generate_real_samples(self, dataset, n_samples):
        pass

    @abstractmethod
    def generate_latent_points(self, latent_dim, n_samples):
        pass

    @abstractmethod
    def generate_fake_samples(self, latent_dim: int, n_samples: int) -> Tuple[np.array, np.array]:
        pass

    def epoch_end_event(self, epoch_number, data_dict):
        for event in self.on_epoch_end:
            event(self, epoch_number, data_dict)

    def train(self, dataset: np.array, n_epochs=100, n_batch=256) -> None:
        bat_per_epo = int(dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        gan_loss = None
        d_loss = None
        for epoch_number in range(n_epochs):
            for batch_number in range(bat_per_epo):
                X_real, y_real = self.generate_real_samples(dataset, half_batch)
                X_fake, y_fake = self.generate_fake_samples(self.latent_dim, half_batch)
                X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
                d_loss, _ = self.discriminator.train_on_batch(X, y)
                X_gan = self.generate_latent_points(self.latent_dim, n_batch)
                y_gan = np.ones((n_batch, 1))
                gan_loss = self.gan.train_on_batch(X_gan, y_gan)
                # Logging the losses to tensorboard
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (epoch_number + 1, batch_number + 1, bat_per_epo, d_loss, gan_loss))
            self.epoch_end_event(epoch_number, {'gan_loss': gan_loss, 'd_loss': d_loss})




