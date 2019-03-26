import pickle
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Dropout, Flatten
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

with open('labeldict.pkl', 'rb') as f:
    labeldict = pickle.load(f)

data = np.load('imagesarray.npz')
X = data['X']
Y = data['Y']
n_classes = len(labeldict)
input_shape = X[0].shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.summary()

EPOCHS = 5
H = model.fit(X_train, Y_train, batch_size=32,
                epochs=EPOCHS, verbose=1, validation_split=0.1)

model.save('gestureModel.hdf5')

score = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy : ", score)
