import io
import tensorflow as tf

from keras.datasets.cifar10 import load_data

from dcgan.cifar10_gan import Cifar10Gan

# Load training data
(trainX, trainy), (testX, testy) = load_data()
trainX = trainX.astype('float32')
trainX = (trainX - 127.5) / 127.5

from matplotlib import pyplot as plt


def plot_to_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def image_grid(examples, n=5):
    examples = (examples + 1) / 2.0
    figure = plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    return figure


def add_plot(examples, n=5):
    image = image_grid(examples, n)
    return image


def tb_epoch_end(gan, epoch_number, data_dict):
    print(f'Epoch {epoch_number}: {data_dict}')
    n = 5
    examples = gan.generator.predict(gan.generate_latent_points(100, n * n))
    image = add_plot(examples, n)
    # plt.show()
    image = plot_to_image(image)
    plt.clf()
    if epoch_number % 10 == 1:
        gan.generator.save_weights(f'./trained_models/cifargen_{epoch_number}.keras')


# Main function
if __name__ == '__main__':
    dcgan = Cifar10Gan()
    dcgan.on_epoch_end.append(tb_epoch_end)
    # tb_epoch_end(dcgan, 0, {})
    # tb_epoch_end(dcgan, 10, {})
    dcgan.train(trainX)
