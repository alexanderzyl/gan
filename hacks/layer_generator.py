import numpy as np
from keras.models import Sequential
from keras.layers import Conv2DTranspose, LeakyReLU, BatchNormalization
from keras.initializers import RandomNormal
from keras.optimizers.legacy import Adam

# define the generator model
model = Sequential()
init = RandomNormal(stddev=0.02)

# upsample using strided convolutions
model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', input_shape=(64, 64, 3), kernel_initializer=init))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.summary()

# compile model
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
model.compile(loss='binary_crossentropy', optimizer=optimizer)


# generate gauss latent points
def generate_latent_points(latent_dim, n_samples):
    return np.random.randn(latent_dim * n_samples).reshape(n_samples, latent_dim)


samples = generate_latent_points(100, 25)
print(samples.shape, samples.mean(), samples.std())


# noisy labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = np.random.choice(a=y.shape[0], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y


y = np.zeros((100, 1))
y = noisy_labels(y, 0.05)
print(y.sum())
