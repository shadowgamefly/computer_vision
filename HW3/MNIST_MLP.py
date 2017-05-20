# Jerry Sun ys7va
# Final version for Project III Part I

from __future__ import print_function

import keras
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

# transform a 2-D array into 1-D pick the highest probablility
# as the output
def transform(twoD):
    length = twoD.shape[0]
    oneD = np.zeros(length)
    for i in range(10000):
        max = -1
        val = 0
        for j in range(10):
            if twoD[i][j] > val:
                max = j
                val = twoD[i][j]
        oneD[i] = max
    return oneD

# normalize the 2-D array by row
def normalize(m):
    for i in range(m.shape[0]):
        sum = 0
        for j in range(m.shape[1]):
            sum = sum + m[i][j]
        for j in range(m.shape[1]):
            m[i][j] = m[i][j]/sum
    return m

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# origin version
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))

# network for Q1
# model.add(Dense(10, activation='softmax', input_shape=(784,)))

# final version
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
result = model.predict(x_test, batch_size=32, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# transform 2-D prediction into 1-D array
y_pred = transform(result)
y_test = transform(y_test)

# create confusion matrix based on two ground_truth and prediction
confusionM = confusion_matrix(y_test, y_pred)

# cast int8 np array into float format
normal = np.float64(confusionM)

# set output np precision
np.set_printoptions(precision=4)

final = normalize(normal)
print(final)
