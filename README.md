## Image Inpainting via Sparse Reconstruction
This code is submission for assginment 1 for the course [EE698K: Modeling and Representation Techiques for Images](http://home.iitk.ac.in/~tanaya/ee698K.html).

* **Requirements**
    * python, g++
    * OpenCV
    * skimage, numpy, scipy

* **To Compile**  
    From the root folder run the following commands
    * `mkdir build`
    * `cd build`
    * `cmake ..`
    * `make`

* **To Create Mask**
    * `./create_mask PATH_TO_IMAGE`

* **To Run Sparse Inpaint**
    * `./sparse_inpaint METHOD VALUE DICT_SIZE`  
    METHOD = IRLS or OMP  
    VALUE  = *epsilon* for IRLS and *spasity* for OMP
    DICT_SIZE = 128 or 256