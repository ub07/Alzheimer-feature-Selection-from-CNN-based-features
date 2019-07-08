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

## required for efficient GPU use
from keras.backend import tensorflow_backend
config = tf.compat.v1.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
## required for efficient GPU use

X_train = np.load("AD/CN Data.npy")
Y_train = np.load("Train_Output.npy")
X_test = np.load("MCI(dataset).npy")
Y_test = np.load("MCI(Output).npy")



#Start Neural Network
model = create_model()
sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
model.compile(loss='binary_crossentropy',
                        optimizer=sgd,
                        metrics = ["accuracy"])

history= model.fit(X_train, Y_train,
         batch_size=batch_size,
         epochs=epochs,
         shuffle=True,        
                  )
#Neural Network ends here

def pre_trained_model():
  pmodel = Sequential()
  pmodel.add(Conv2D(32, kernel_size=(5,5), padding="same",input_shape=(32,32,3),weights = model.layers[0].get_weights()))
  pmodel.add(BatchNormalization(weights = model.layers[1].get_weights()))
  pmodel.add(Activation('relu'))
  pmodel.add(MaxPooling2D(pool_size=(3,3),strides = 2,padding = "same"))
  pmodel.add(Conv2D(32, kernel_size=(5,5), padding="same",weights = model.layers[4].get_weights()))
  pmodel.add(BatchNormalization(weights = model.layers[5].get_weights()))
  pmodel.add(Activation('relu'))          
  pmodel.add(AveragePooling2D(pool_size=(3,3),strides = 2,padding = "same"))
  pmodel.add(Conv2D(64, kernel_size=(5,5), padding="same",weights = model.layers[8].get_weights()))
  pmodel.add(BatchNormalization(weights = model.layers[9].get_weights()))
  pmodel.add(Activation('relu'))          
  pmodel.add(AveragePooling2D(pool_size=(3,3),strides = 2, padding = "same"))
  pmodel.add(Flatten())
  return pmodel


pmodel = pre_trained_model()
sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
pmodel.compile(loss='mean_squared_error',
                          optimizer=sgd,
                          metrics = ["accuracy"])
result = pmodel.predict(X_test)
np.save("features.npy",result)



  
  
  
