import numpy as np
import cv2

cap = cv2.VideoCapture('bunny1.mp4')
if (!cap.isOpened()):
    print "cannot open file!"
    exit(1)


while(cap.isOpened()):
    ret, frame = cap.read()
    print frame

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
