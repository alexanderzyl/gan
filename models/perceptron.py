from keras import Model
from keras.layers import Input, Dense
from keras.src.utils import plot_model

visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
output = Dense(10, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)

model.summary()
plot_model(model, to_file='multilayer_perceptron_graph.png')
