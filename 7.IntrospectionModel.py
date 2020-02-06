from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

from Utility.StandartDatasets import get_mnist_dataset, get_cifar10_dataset
from Utility.ResudialModel import get_model

batch_size = 128
num_classes = 10
epochs = 20
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = get_cifar10_dataset()

model = get_model(x_train.shape[1:], num_classes)

for layer in model.layers:
    print(f'class: {layer.__class__.__name__}')
    print(f'layer name:{layer.name}')
    print(f'input shape:{layer.input_shape}')
    print(f'output shape:{layer.output_shape}')

    config = layer.get_config()

    print(f'config:{config}')
    print('')

