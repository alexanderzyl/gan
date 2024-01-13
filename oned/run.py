import numpy as np
from keras.layers import Input, Dense
from keras.models import Model


# create the discriminator model
def create_discriminator(n_inputs=2):
    inp = Input(shape=(n_inputs,))
    x = Dense(25, activation='relu')(inp)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def real_function(x):
    return x * x


def generate_real_samples(n):
    X1 = np.random.rand(n) - 0.5
    X2 = real_function(X1)
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y


def generate_fake_samples(n):
    X1 = np.random.rand(n) - 0.5
    X2 = np.random.rand(n) - 0.5
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    y = np.zeros((n, 1))
    return X, y


def train_discriminator(model, n_epochs=1000, n_batch=128):
    for i in range(n_epochs):
        X_real, y_real = generate_real_samples(n_batch)
        model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(n_batch)
        model.train_on_batch(X_fake, y_fake)
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
        print(i, acc_real, acc_fake)


model = create_discriminator()
model.summary()

train_discriminator(model)

# predict
X, y = generate_real_samples(100)
y_pred = model.predict(X)
print(np.count_nonzero(y_pred > 0.5)/y_pred.shape[0])

X, y = generate_fake_samples(100)
y_pred = model.predict(X)
print(np.count_nonzero(y_pred < 0.5)/y_pred.shape[0])

