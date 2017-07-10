from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
vid = cv2.VideoCapture('test.avi')
ret,frame = vid.read()

while ret:
    ret,frame = vid.read()
    frame = imutils.resize(frame,width = min(400, frame.shape[1]))
    orignal = frame.copy()

    # DETECTING PEOPLE
    (rects,weights) = hog.detectMultiScale(frame,winStride = (4, 4), padding = (8, 8), scale = .05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orignal, (x,y), (x+w, y+h), (0, 0, 255), 2)
    
    # Now applying non maxima supression with fairely large threshhold to 
    # maintain overlapping that are still people
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick  = non_max_suppression(rects, probs = None, overlapThresh = 0.65)
    
    # Drawing final boundary boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0,255,0), 2)

    print("[INFO] {} orignal boxes, {} after supression".format(len(rects), len(pick)))

    # show Output
    cv2.imshow("Before NMS", orignal)
    cv2.imshow("After NMS", frame)
    cv2.waitKey(1)
