from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

visible = Input(shape=(64, 64, 1))
conv1 = Conv2D(32, (4, 4), activation='relu')(visible)
pool1 = MaxPooling2D()(conv1)

conv2 = Conv2D(16, (4, 4), activation='relu')(pool1)
pool2 = MaxPooling2D()(conv2)

flat1 = Flatten()(pool2)
hidden1 = Dense(10, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(hidden1)

model = Model(inputs=visible, outputs=output)
model.summary()

# plot_model(model, to_file='convolution_graph.png')