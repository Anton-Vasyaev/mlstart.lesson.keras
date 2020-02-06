from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

def resNetBlock(input, filters, k, batchNorm=False):
    x1 = Conv2D(int(filters / 2), kernel_size=k, padding='SAME')(input)
    if batchNorm == True:
        x1 = BatchNormalization()(x1)

    x1 = Activation('relu')(x1)

    x = Conv2D(filters, kernel_size=k, padding='SAME')(x1)
    if batchNorm == True:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=k, padding='SAME')(x)
    if batchNorm == True:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(int(filters / 2), kernel_size=k, padding='SAME')(x)
    if batchNorm == True:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x1, x])

    return x


def getDense(input, size):
    x = Dense(size)(input)
    x = Activation('relu')(x)

    return x


def get_model(input_shape, num_classes, batchNorm=False):
    input_layer = Input(shape=input_shape)

    x = resNetBlock(input_layer, 128, (3, 3), batchNorm)    # -> 32x32

    x = MaxPooling2D((2, 2))(x)                             # -> 16x16

    x = resNetBlock(x, 256, (3, 3), batchNorm)              # -> 16x16

    x = MaxPooling2D((2, 2))(x)                             # -> 8x8

    x = resNetBlock(x, 64, (3, 3), batchNorm)               # -> 8x8

    x = MaxPooling2D((2, 2))(x)                             # -> 4x4

    x = Flatten()(x)

    x = getDense(x, 128)

    x = Dense(num_classes)(x)
    output_layer = Activation('softmax')(x)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model
