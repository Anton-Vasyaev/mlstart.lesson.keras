from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Flatten
from tensorflow.keras import backend as K

from Utility.StandartDatasets import get_mnist_dataset

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

batch_size = 128
num_classes = 10
epochs = 12

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = get_mnist_dataset()

model = Sequential()

model.add(Input(shape=x_train.shape[1:]))
model.add(Flatten())

model.add(Dense(512))
model.add(Activation('sigmoid'))

model.add(Dense(512))
model.add(Activation('sigmoid'))


model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])