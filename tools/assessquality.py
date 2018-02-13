import niqe
import scipy.misc
import numpy
import glob

image_path = "../images/"
results = glob.glob(image_path+"results/*.JPG")

original  = scipy.misc.imread (image_path + "original.JPG").astype(numpy.float32)

print "NIQE Value for the original image is :" + str(niqe.niqe(original/255))

for name in results:
    im = scipy.misc.imread (name).astype(numpy.float32)
    value = niqe.niqe(im/255)
    print "NIQE Value of the image " + name + "is : " + str(value)
