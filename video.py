from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import os
import time
count = 0
if not os.path.isdir("Captures"):
    os.mkdir("Captures")
vs = cv2.VideoCapture(0)
while True:
    cap, frame = vs.read()
    count = count + 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if count % 15 == 0:
        sec = time.time()
        cv2.imwrite("Captures/im"+str(count)+"_"+str(sec)+".jpg", frame)

    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()