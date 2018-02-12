import subprocess
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import sys
sys.path.insert(0, './code/tools/src/')
from niqe import niqe

ep_start = 0
ep_end = -5
N = 20

epsilon = np.logspace(start=ep_start, stop=ep_end, num=N)
print epsilon

### For Dictionary of 128 size and OMP

omp128 = np.zeros (epsilon.shape)

count = 0
for ep in epsilon:
    p = subprocess.Popen(["./code/app/src/main.out", str(ep), "128"],
                         stdout=subprocess.PIPE)
    
    while p.poll() == None:
        time.sleep (0.1)
        
    img = scipy.misc.imread("./data/" + str(ep) + "_128_omp.jpg",          
                            flatten=True).astype(np.float)/255.0
    omp128[count] = niqe(img)
    
    count += 1
    
print omp128

### For Dictionary of 256 size and OMP

omp256 = np.zeros (epsilon.shape)

count = 0
for ep in epsilon:
    p = subprocess.Popen(["./code/app/src/main.out", str(ep), "256"],
                         stdout=subprocess.PIPE)
    
    while p.poll() == None:
        time.sleep (0.1)
        
    img = scipy.misc.imread("./data/" + str(ep) + "_256_omp.jpg",          
                            flatten=True).astype(np.float)/255.0
    omp256[count] = niqe(img)
    
    count += 1
    
print omp256

### For Dictionary of 128 size and IRLS

irls128 = np.zeros (epsilon.shape)

count = 0
for ep in epsilon:
    p = subprocess.Popen(["./code/app/src/main.out", str(ep), "128", "irls"],
                         stdout=subprocess.PIPE)
    
    while p.poll() == None:
        time.sleep (0.1)
        
    img = scipy.misc.imread("./data/" + str(ep) + "_128_irls.jpg",          
                            flatten=True).astype(np.float)/255.0
    irls128[count] = niqe(img)
    
    count += 1
    
print irls128

### For Dictionary of 256 size and IRLS

irls256 = np.zeros (epsilon.shape)

count = 0
for ep in epsilon:
    p = subprocess.Popen(["./code/app/src/main.out", str(ep), "256", "irls"],
                         stdout=subprocess.PIPE)
    
    while p.poll() == None:
        time.sleep (0.1)
        
    img = scipy.misc.imread("./data/" + str(ep) + "_256_irls.jpg",          
                            flatten=True).astype(np.float)/255.0
    irls256[count] = niqe(img)
    
    count += 1
    
print irls256

plt.grid(True, which="both")
plt.semilogx()

plt.plot(epsilon, omp128,  label = "OMP with 128 atoms")
plt.plot(epsilon, omp256,  label = "OMP with 256 atoms")
plt.plot(epsilon, irls128, label = "IRLS with 128 atoms")
plt.plot(epsilon, irls256, label = "IRLS with 256 atoms")
plt.legend(loc = "upper left", title = "Legend", fancybox=True)

plt.title ('NIQE vs Error Threshold')
plt.xlabel('Error Threshold')
plt.ylabel('NIQE')

plt.show()