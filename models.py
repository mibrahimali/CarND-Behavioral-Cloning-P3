from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D

def lenet_model(input_shape):
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    model.add(Convolution2D(32, 5, 5, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def nvidia_model(input_shape):

    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5 - 1.00), input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='he_normal', activation='elu'))
    # model.add(Dropout(0.1))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='he_normal', activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='he_normal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal', activation='elu'))
    # model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu', init='he_normal'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu', init='he_normal'))
    model.add(Dropout(0.5))
    # model.add(Dense(50, activation='elu', init='he_normal'))
    model.add(Dense(10, activation='elu', init='he_normal'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, init='he_normal'))
    return model