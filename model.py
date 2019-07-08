import tensorflow as tf
import numpy as np

if tf.__version__.split('.')[1] == '13':
    raise Exception('This notebook is not compatible with Tensorflow 1.13, please use the previous version at https://github.com/tensorflow/tpu/blob/913cf31d85bc31541fbdafa9d0b87db71dd6dcba/tools/colab/fashion_mnist.ipynb')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import optimizers



#size of parameters
batch_size = 151
num_classes = 2
epochs = 30
momentum = 0.9
lr = 0.001
decay = 0.0001



def create_model():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(5,5), padding="same",input_shape=(32,32,3)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3,3),strides = 2,padding = "same"))
  model.add(Conv2D(32, kernel_size=(5,5), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))          
  model.add(AveragePooling2D(pool_size=(3,3),strides = 2,padding = "same"))
  model.add(Conv2D(64, kernel_size=(5,5), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))          
  model.add(AveragePooling2D(pool_size=(3,3),strides = 2, padding = "same"))
  model.add(Flatten())
  model.add(Dense(num_classes,use_bias=False))
  model.add(Activation('softmax'))
  return model
