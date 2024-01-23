import numpy as np

from dcgan.mnist_gan import MnistGan
import matplotlib.pyplot as plt


class DemoMnistGan(MnistGan):
    def __init__(self):
        super().__init__()
        self.generator.load_weights('../trained_models/generator_190.keras')
        self.discriminator.load_weights('../trained_models/discriminator_190.keras')
        self.create_gan()

    def generate_latent_points(self, latent_dim, n_samples):
        latents = np.zeros((n_samples, latent_dim))
        # for i in range(n_samples):
        #     latents[i, :] = i * 0.1

        # latents[:, 13:20] = 1.5

        return latents


def generate_image():
    gan = DemoMnistGan()
    images, _ = gan.generate_fake_samples(9)
    # from [-1,1] to [255,0]
    images = (1. - images) / 2.0 * 255.0
    images = images.astype(np.uint8)
    for i in range(9):
        plt.subplot(3, 3, 1 + i)
        plt.axis('off')
        plt.imshow(images[i, :, :, 0], cmap='gray_r')
    plt.show()


generate_image()
