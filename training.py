from keras.datasets import mnist
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.utils import np_utils

LEARNING_RATE = 0.001

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

model = Sequential()
model.add(layers.Conv2D(16, [4, 4], input_shape=[28, 28, 1]))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, [2, 2]))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(10))

model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=1)

model.save("model.h5")

