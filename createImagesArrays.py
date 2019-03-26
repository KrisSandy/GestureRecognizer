import cv2
import numpy as np
import glob
import os
import re
import pickle

def getName(imgpath):
    name = os.path.basename(filename)[:-4]
    return " ".join(re.findall("[a-z]+", name))

label_dict = dict()
label_count = 0
image_shape = None

for filename in glob.glob('./images/*.png'):
    name = getName(filename)
    if image_shape == None:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        image_shape = (h, w, 1)
    if name not in label_dict:
        label_dict[name] = label_count
        label_count += 1

X = []
Y = []
n_classes = len(label_dict)
for filename in glob.glob('./images/*.png'):
    X.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE).reshape(image_shape))
    Y.append(label_dict[getName(filename)])

X = np.array(X)
Y = np.array(Y)
X = X.astype('float32')
X /= 255 

print(X.shape)
print(Y.shape)

np.savez('imagesarray', X=X, Y=Y)

with open('labeldict.pkl', 'wb') as f:
        pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)
