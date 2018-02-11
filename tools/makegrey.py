import cv2
import glob
import numpy as np
import matplotlib as plt

def displayImage (frame, frame_name='rekt'):
    while 1:
        cv2.imshow (str(frame_name), frame)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

dir = '../images/color/'
imagelist = glob.glob(dir+'*.JPG')

for name in imagelist:
    im = cv2.imread(name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    name = name.replace("color", "gray")
    print name
    cv2.imwrite(name, im)