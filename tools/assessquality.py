import niqe
import scipy.misc
import numpy

image_path = "../images/"

irls128 = scipy.misc.imread (image_path + "results/irls128.JPG").astype(numpy.float32)
irls256 = scipy.misc.imread (image_path + "results/irls256.JPG").astype(numpy.float32)
omp128  = scipy.misc.imread (image_path + "results/omp128.JPG").astype(numpy.float32)
omp256  = scipy.misc.imread (image_path + "results/omp256.JPG").astype(numpy.float32)

original  = scipy.misc.imread (image_path + "original.JPG").astype(numpy.float32)

print "NIQE Values for inpainted images are as :"
print "IRLS, Dictionary Size = 128: " + str(niqe.niqe(irls128/255))
print "IRLS, Dictionary Size = 256: " + str(niqe.niqe(irls256/255))
print "OMP, Dictionary Size = 128: " + str(niqe.niqe(omp128/255))
print "OMP, Dictionary Size = 256: " + str(niqe.niqe(omp256/255))

print "NIQE Value for the original image is :" + str(niqe.niqe(original/255))