import pickle
import cv2
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Dropout, Flatten
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from globalSettings import *
from globalUtils import extractROI

cap = cv2.VideoCapture(0)

# load model
model1 = load_model('gestureModel.hdf5')

# load labels dictionary
with open('labeldict.pkl', 'rb') as f:
    labeldict = pickle.load(f)

# preconfigure plot
plt.ion()
fig = plt.figure()
subp = fig.add_subplot(111)
subp.set_ylim(0, 1)
bars = subp.bar(labeldict.keys(),np.zeros(len(labeldict)))

while True:
    _, frame = cap.read()

    # flip to compensate mirror effect
    frame = cv2.flip(frame, 1)
    
    # reduce size of image
    frame = cv2.resize(frame, (640,480))
    
    # extract region of interest and convert to binary
    final_roi = extractROI(frame, x, y, h, w)

    # display stream
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),1)
    cv2.imshow('roi', final_roi)
    cv2.imshow('frame', frame)

    # make prediction
    X = np.array(final_roi).astype('float32')
    X = X.reshape(h, w, 1)
    X /= 255
    prediction = model1.predict(np.array([X]))
    print(prediction)

    # update plot
    [bar.set_height(h) for bar,h in zip(bars,prediction.reshape(3))]
    fig.canvas.draw()
    fig.canvas.flush_events()

    # process key
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()