from numpy import asarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose

model = Sequential()
trans_layer = Conv2DTranspose(1, (1, 1), strides=(2, 2), input_shape=(2, 2, 1), padding='same')
model.add(trans_layer)
model.set_weights([asarray([[[[1.]]]]), asarray([0.0])])
model.summary()

X = asarray([1, 2, 3, 4])
X = X.reshape((1, 2, 2, 1))
print(X)
yhat = model.predict(X)
print(yhat.reshape((4, 4)))
