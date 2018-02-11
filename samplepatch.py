import cv2
import glob
import numpy as np
from random import randint

#Read Images from directory
imlist = glob.glob('./images/gray/*.JPG')
dictdir = './images/dictionary/'

#Constants
PATCHSIZE = [8, 8]
DICTIONARYSIZE = 256
IMAGESIZE = [1280, 960]

#Stacked Image contains all of the images for uniform sampling
stacked_image = np.empty((IMAGESIZE), dtype=np.uint8)

for name in imlist:
    im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    stacked_image = np.concatenate((stacked_image, im), axis=1)
    print stacked_image.shape

#Remove the first empty part form stacked image
stacked_image = stacked_image[:, (IMAGESIZE[1]-1):]
print stacked_image.shape

#temp_image is to concatenate the other portions
temp_image = np.zeros((64, 1), dtype=np.uint8)
for i in range(DICTIONARYSIZE):
    topleft = [randint(0, stacked_image.shape[0]-9), randint(0, stacked_image.shape[1]-9)]
    patch = stacked_image[topleft[0]:(topleft[0] + PATCHSIZE[0]), topleft[1]:(topleft[1] + PATCHSIZE[0])]
    cv2.imwrite(dictdir + str(i)+'.JPG', patch)
    patch = np.reshape(patch, (64, 1))
    temp_image = np.concatenate((temp_image, patch), axis=1)

#Remove the empty frame from dictionary
print temp_image.shape
dictionary = temp_image[:, 1:]
print dictionary.shape

cv2.imwrite("dictionary.JPG", dictionary)