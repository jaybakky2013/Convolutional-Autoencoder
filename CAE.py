from keras.layers import Flatten, Input, Conv2D, BatchNormalization, Dense, ZeroPadding2D, MaxPooling2D, \
    Conv2DTranspose, Reshape
from keras.models import Model
from keras import optimizers
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os


def create_model(shape):
    a = 'tanh'
    input_tensor = Input(shape=shape.shape)
    x1 = ZeroPadding2D((1, 1))(input_tensor)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=a, name='E1')(x1)
    x2 = MaxPooling2D((2, 2), strides=(1, 1))(x2)
    x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=a, name='E2')(x2)
    x3 = MaxPooling2D((2, 2), strides=(1, 1))(x3)
    x4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=a, name='E3')(x3)
    x4 = MaxPooling2D((2, 2), strides=(1, 1))(x4)
    x5 = Flatten()(x4)
    x5 = Dense(256)(x5)
    x5 = Reshape((8, 8, 4))(x5)
    y1 = Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=a, name='D1')(x5)
    y1 = BatchNormalization()(y1)
    y2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=a, name='D2')(y1)
    y2 = BatchNormalization()(y2)
    y3 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation=a, name='D3')(y2)
    y3 = BatchNormalization()(y3)
    y4 = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=a, name='D4')(y3)
    y4 = BatchNormalization()(y4)
    model = Model(inputs=input_tensor, outputs=y4)
    return model


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def normalization(image, min_in, max_in, min_out, max_out):
    scale = np.array((image - min_in) / float(max_in - min_in), dtype=float)
    return min_out + (scale * (max_out - min_out))


def generate_arrays_from_file(files, batch_size):
    for item in files:
        d = unpickle('./data/' + item)
        d = np.split(d, len(d) // batch_size)
        for item in d:
            temp = item.reshape((len(item), 3, 64, 64)).transpose(0, 2, 3, 1)
            temp = normalization(temp, 0, 255, -1, 1)
            yield (temp, temp)


if __name__ == '__main__':
    coefficient_of_usage = 10
    batch_size = 10
    epochs = 10
    split = 0.8
    lr_ = 1e-6 * coefficient_of_usage * batch_size
    total_train = 160000
    total_test = 40000


    shape = np.array(np.array((range(12288)))).reshape((64, 64, 3))
    autoencoder = create_model(shape)
    opt = optimizers.adam(lr=lr_)
    autoencoder.compile(loss='mean_squared_error',
                        optimizer=opt,
                        metrics=['accuracy'])

    print(os.listdir('./data'))
    files = os.listdir('./data')
    threshold = int(len(files) * split)
    train = files[:threshold]
    test = files[threshold:]
    autoencoder_train = autoencoder.fit_generator(generate_arrays_from_file(train, batch_size),
                                                  validation_data=generate_arrays_from_file(test, batch_size),
                                                  steps_per_epoch=(total_train // coefficient_of_usage) // batch_size,
                                                  validation_steps=(total_test // coefficient_of_usage) // batch_size,
                                                  epochs=epochs)

    loss = autoencoder_train.history['loss']
    print(loss)
    loss = loss
    val_loss = autoencoder_train.history['val_loss']
    print(val_loss)
    val_loss = val_loss
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()