import cv2
import os
import shutil
from globalSettings import *
from globalUtils import extractROI

cap = cv2.VideoCapture(0)
saveimg = False
imgcounter = 0
maximages = 0
path = "./images/"

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
    cv2.imshow('Stream', frame)
    cv2.imshow('roi', final_roi)

    # save images with gesture name
    if saveimg:
        if imgcounter < maximages:
            imgcounter += 1
            name = gestureName + str(imgcounter) + ".png"
            print('saving image ', name)
            cv2.imwrite(path+name, final_roi)
        else:
            saveimg = False
            imgcounter = 0
            maximages = 0

    key = cv2.waitKey(1)

    # process keys
    if key == ord('q'):
        break    
    elif key == ord('r'):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        print("Path reset complete...")
    elif key == ord('n'):
        gestureName = input('Enter gesture name : ')
        maximages = int(input('Enter collection size : '))
        print("When ready press key 'c' to start capturing")
    elif key == ord('c'):
        if maximages == 0:
            print("Set the capture settings first by pressing key 'n'...")
        else:
            saveimg = True

cap.release()
cv2.destroyAllWindows()
