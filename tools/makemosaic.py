import cv2
import numpy as np

imdir = "../images/dictionary/"

dictionary128 = np.zeros((16*8, 8*8), dtype=np.uint8)
dictionary256 = np.zeros((16*8, 16*8), dtype=np.uint8)

x = 0
y = 0

for i in range(256):
    x = i % 16
    y = i / 16
    im = cv2.imread (imdir + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
    if (y < 8):
            dictionary128[8*x:(8*(x + 1)), 8*y:(8*(y + 1))] = im

    dictionary256[8*x:(8*(x + 1)), 8*y:(8*(y + 1))] = im

dictionary128 = cv2.resize (dictionary128, (0, 0),  fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
dictionary256 = cv2.resize (dictionary256, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


cv2.imwrite(imdir + "dictionary128.png", dictionary128)
cv2.imwrite(imdir + "dictionary256.png", dictionary256)
