from keras.layers import Flatten, Input, Conv2D, BatchNormalization, Dense, ZeroPadding2D, MaxPooling2D, \
    Conv2DTranspose, Reshape
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import Model
from keras import optimizers
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from keras import backend as K
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def model(X):
    X = unpickle(X)
    K.clear_session()
    input_ = Input(shape=(X.shape[1],))

    layer_1 = Dense(2000, activation="tanh")(input_)
    layer_2 = Dense(1000, activation="tanh")(layer_1)
    layer_3 = Dense(500, activation="tanh")(layer_2)
    encoding = Dense(250, activation=None)(layer_3)
    layer_5 = Dense(1000, activation="tanh")(encoding)
    layer_6 = Dense(1000, activation="tanh")(layer_5)
    layer_7 = Dense(2000, activation='tanh')(layer_6)
    decoded = Dense(X.shape[1])(layer_7)

    return Model(inputs=input_, outputs=decoded)


def normalization(image, min_in, max_in, min_out, max_out):
    scale = np.array((image - min_in) / float(max_in - min_in), dtype=float)
    return min_out + (scale * (max_out - min_out))


def generate_arrays_from_file(files, batch_size):
    for item in files:
        d = unpickle('./data/' + item)
        d = np.split(d, len(d) // batch_size)
        for item in d:
            temp = item.reshape(len(item), 12288,)
            temp = normalization(temp, 0, 255, -1, 1)
            yield (temp, temp)

if __name__ == '__main__':

    EPOCHS = 60
    BATCH_SIZE = 10
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'mean_squared_error'
    ACTIVATION_FUNCTION = 'tanh'  # the last activation needs to be softmax.

    # how many nodes for each layer.
    # decoder
    INPUT_LAYER = 12288
    ENCODER_LAYER_1 = 1024
    ENCODER_LAYER_2 = 512
    ENCODER_LAYER_3 = 256
    # encoder
    DECODER_LAYER_1 = 512
    DECODER_LAYER_2 = 1024
    DECODER_LAYER_3 = 12288

    split = 0.8


    # t0 = unpickle('./data/train_00')
    # t1 = unpickle('./data/train_01')
    # data = np.concatenate((t0, t1), axis=0)

    files = os.listdir('./data')
    threshold = int(len(files) * split)
    train = files[:threshold]
    test = files[threshold:]


    # ind = int(len(data) * split)
    # x_train, x_test = data[:ind], data[ind:]
    # x_train, x_test = x_train, x_test
    # x_train = normalization(x_train, 0, 255, -1, 1)
    # x_test = normalization(x_test, 0, 255, -1, 1)

    # print(data.shape)
    # print(x_train.shape)
    # print(x_test.shape)

    # autoencoder = model(x_train)
    autoencoder = model(train[0])
    opt = optimizers.adam(lr=1e-4)
    autoencoder.compile(loss='mean_squared_error',
                        optimizer=opt,
                        metrics=['accuracy'])

    autoencoder_train = autoencoder.fit_generator(generate_arrays_from_file(train, BATCH_SIZE),
                                                  validation_data=generate_arrays_from_file(test, BATCH_SIZE),
                                                  steps_per_epoch=160000 // BATCH_SIZE,
                                                  validation_steps=40000 // BATCH_SIZE,
                                                  epochs=EPOCHS)

    # autoencoder_train = autoencoder.fit(x_train, x_train,
    #                                     batch_size=BATCH_SIZE,
    #                                     epochs=EPOCHS,
    #                                     validation_data=(x_test, x_test))

    loss = autoencoder_train.history['loss']
    loss = loss
    val_loss = autoencoder_train.history['val_loss']
    val_loss = val_loss
    epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()