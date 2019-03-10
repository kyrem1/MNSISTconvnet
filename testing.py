from keras.datasets import mnist
from keras.utils import np_utils
from keras import models

from keras.utils import plot_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

num_classes = 10

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

model = models.load_model('model.h5')
print(model.evaluate(x_test, y_test, verbose=1))

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
