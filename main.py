import numpy as np
print("Imported NumPy")
from keras.models import Sequential
print("Imported Keras Model")
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
print("Imported Keras Layers")
from keras.utils import np_utils
print("Imported Keras Utilities")
from keras.datasets import mnist
print("Imported MNIST")
import keras
print("Imported Keras")

print("Program started")

np.random.seed(123)

#Load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Preprocess data, reshaping to a depth of 1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print (X_train.shape)

#Scale values between 0 and 1 for all pixels, white = 1, black = 0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Reshape data labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print (Y_train.shape)

print("Initiation completed")

try:
    model = keras.models.load_model("MNIST_Model.h5")
except:
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

print (model.output_shape)

model.fit(X_train, Y_train,
          batch_size=128,
          epochs=2,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("Program completed")

model.save("MNIST_Model.h5")
print("Model saved")
