import numpy as np
import tensorflow as tf
from keras.datasets.mnist import load_data
from tensorflow.keras.callbacks import TensorBoard
from dcgan.mnist_gan import MnistGan

# Load training data
(trainX, trainy), (testX, testy) = load_data()
trainX = trainX.astype('float32')
trainX = (trainX - 127.5) / 127.5
trainX = np.expand_dims(trainX, axis=-1)

# Adding TensorBoard
# %reload_ext tensorboard
# !rm -rf logs
# %tensorboard --logdir logs
log_dir = './logs'
tensorboard_gan = TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
file_writer = tf.summary.create_file_writer(log_dir)


def generate_images(gan, num_rows=1, num_cols=1):
    g_model = gan.generator
    latent_dim = gan.latent_dim
    num_images = num_rows * num_cols
    generated_images, _ = gan.generate_fake_samples(g_model, latent_dim, num_images)
    # rescale the pixel values from [-1,1] to [0,255] to display images
    generated_images = (generated_images + 1.) / 2. * 255.
    generated_images.reshape(num_images, 28, 28)
    # Initialize an empty array to hold the final large image
    final_image = np.zeros((num_rows * 28, num_cols * 28))

    image_index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            final_image[row * 28: (row + 1) * 28, col * 28: (col + 1) * 28] = generated_images[
                image_index].squeeze()
            image_index += 1

    return final_image


def tb_epoch_end(gan, epoch_number, data_dict):
    with file_writer.as_default():
        generated_image = generate_images(gan, num_rows=5, num_cols=5)
        generated_image = np.reshape(generated_image,
                                     (1, generated_image.shape[0], generated_image.shape[1], 1))
        tf.summary.image("Generated image", generated_image, step=epoch_number)
    tensorboard_gan.on_epoch_end(epoch_number, data_dict)
    if epoch_number % 10 == 0:
        gan.generator.save_weights(f'./models/generator_{epoch_number}.keras')
        gan.discriminator.save_weights(f'./models/discriminator_{epoch_number}.keras')


# Main function
if __name__ == '__main__':
    dcgan = MnistGan()
    dcgan.on_epoch_end.append(tb_epoch_end)

    # Create a TensorBoard callback
    tensorboard_gan.set_model(dcgan.gan)

    dcgan.train(trainX)
