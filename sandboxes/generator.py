import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose

model = Sequential()
# define input shape, output enough activations for 128 5x5 image

shape = (5, 5, 128)
model.add(Dense(np.prod(shape), input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape(shape))
# double input from 128 5x5 to 10x10 feature maps with transpose convolutions
model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same'))

model.summary()

# generate a random noise vector
x = np.random.randn(100)

# get the output of the model
y = model.predict(x.reshape((1, 100)))

# show the image with matplotlib
import matplotlib.pyplot as plt
plt.imshow(y.reshape((10, 10)), cmap='gray')
plt.show()
