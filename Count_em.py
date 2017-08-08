import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
vid = cv2.VideoCapture('Town.avi')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('People.avi',fourcc, 30.0, (640,360))
#
# out = cv2.VideoWriter('test.avi',cv2.CV_FOURCC('m', 'p', '4', 'v'),120,(640,360))

# create a background subtarctor

bgs = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((11, 11), np.uint8)
areaTH = 200
while vid.isOpened():
    _, frame = vid.read()
    try:
        frame = imutils.resize(frame, height=480, width=min(640, frame.shape[1]))
        orignal = frame.copy()
        bgs_mask = bgs.apply(frame)
        rects = []
        edges = cv2.Canny(frame, 100, 200)
        ret, thresh = cv2.threshold(bgs_mask, 200, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

        cv2.imshow('edges', edges)
        # cv2.imshow('frame',frame)
        # cv2.imshow('Subtracted', bgs_mask)
        # cv2.imshow('Open', opening)
        # cv2.imshow('CLose', closing)

    except:
        print 'Finish'
        break
    _, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        # cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3, 8)
        area = cv2.contourArea(cnt)
        if area > areaTH:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append([x, y, w, h])
            print rects
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print frame.shape
    cv2.imshow('Frame', frame)
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    picks = non_max_suppression(rects, probs=None, overlapThresh=0.5)
    for (xA, yA, xB, yB) in picks:
        cv2.rectangle(orignal, (xA, yA), (xB, yB), (120,160,230))
    cv2.imshow('NMs',orignal)
    # out.write(frame)
    # cv2.imshow('Image', img)
    cv2.waitKey(20)

vid.release()
# out.release()

cv2.destroyAllWindows()
