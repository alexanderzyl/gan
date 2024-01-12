from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

visible = Input(shape=(100,1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)

model.summary()