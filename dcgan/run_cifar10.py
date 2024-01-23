from keras.datasets.cifar10 import load_data

from dcgan.cifar10_gan import Cifar10Gan

# Load training data
(trainX, trainy), (testX, testy) = load_data()
trainX = trainX.astype('float32')
trainX = (trainX - 127.5) / 127.5


def tb_epoch_end(gan, epoch_number, data_dict):
    print(f'Epoch {epoch_number}: {data_dict}')
    if epoch_number % 10 == 0:
        gan.generator.save_weights(f'./trained_models/cifargen_{epoch_number}.keras')


# Main function
if __name__ == '__main__':
    dcgan = Cifar10Gan()
    dcgan.on_epoch_end.append(tb_epoch_end)
    dcgan.train(trainX)
