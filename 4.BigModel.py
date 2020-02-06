from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

from Utility.StandartDatasets import get_mnist_dataset, get_cifar10_dataset

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

batch_size = 128
num_classes = 10
epochs = 20
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = get_cifar10_dataset()

model = Sequential()

model.add(Input(shape=x_train.shape[1:]))

model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Dropout(0.3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])