import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, BatchNormalization
from keras.initializers import RandomNormal
from keras.optimizers.legacy import Adam

# define the discriminator model

model = Sequential()
init = RandomNormal(stddev=0.02)

# downsample using strided convolutions
model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(64, 64, 3), kernel_initializer=init))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.summary()

# compile model
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# scale input data to [-1,1]
def scale_images(images):
    # scale from [0,255] to [-1,1]
    return (images.astype('float32') - 127.5) / 127.5


# label smoothing
def smooth_positive_labels(y):
    v = y - 0.3 + (np.random.random(y.shape) * 0.5)
    return np.clip(v, a_max=1.0, a_min=0.0)


n_samples = 100
y = np.ones((n_samples, 1))
y = smooth_positive_labels(y)
print(y.shape, y.min(), y.max())
