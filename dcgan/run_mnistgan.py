import numpy as np
from keras.datasets.mnist import load_data

from mnist_gan import MnistGan

# Load training data
(trainX, trainy), (testX, testy) = load_data()
trainX = trainX.astype('float32')
trainX = (trainX - 127.5) / 127.5
trainX = np.expand_dims(trainX, axis=-1)


def tb_epoch_end(gan, epoch_number, data_dict):
    print(f'Epoch {epoch_number}: {data_dict}')
    if epoch_number % 10 == 0:
        gan.generator.save_weights(f'./trained_models/generator_{epoch_number}.keras')
        gan.discriminator.save_weights(f'./trained_models/discriminator_{epoch_number}.keras')


# Main function
if __name__ == '__main__':
    dcgan = MnistGan()
    dcgan.on_epoch_end.append(tb_epoch_end)

    dcgan.train(trainX)
