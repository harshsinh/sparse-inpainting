## Image Inpainting via Sparse Reconstruction
This code is submission for assginment 1 for the course [EE698K: Modeling and Representation Techiques for Images](http://home.iitk.ac.in/~tanaya/ee698K.html).

___
* `./code/src`: contains all of the source files, which are  
    * `irls.cxx` and `omp.cxx` : Function Implementation for IRLS and OMP
    * `sparse_inpainting.cxx` : Sparse reconstruction based Inpainting implementation.
    * `main.cxx` : The main driver code for the program
    * `create_mask.cxx` : OpenCv based program that lets you create masks on custom images.
* `./code/include`: contains the corresponding header files.
* `./tools/` : contains various python scripts.
    * `makegrey.py` : Take images stored in `./images/color/` and creates their corresponding grayscale images and stores them in `./images/gray`
    * `samplepatch.py` : Takes the images stored in `./images/gray`, randomly samples 256 different 8x8 patches and stores them in `./images/dictionary/` in a numbered fashion. It also creats an image `./images/dictionary/dictionary.png` whoes columns are the 8x8 patches vectorized.
    * `makemosaic.py` : Takes the images produced from the previous script and generates `./images/dictionary/dictionary128.png` and `./images/dictionary/dictionary256.png`  which have 16x8 and 16x16, 8x8 original patches as their elements.
    * `assessquality.py` : Runs the NIQE assessment on the images inside the `./images/results` folder and displays the results on terminal.

---
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
    In PATH_TO_IMAGE specify the locaiton of source image. Draw using a continuous line.  
    _As of now the limitation on this part is that the mask being drawn has to begin and end close by on the image. This is dut to the fact that I have stored the mask as `contrours` first._

* **To Run Sparse Inpaint**
    * `./sparse_inpaint METHOD VALUE DICT_SIZE`  
    METHOD = IRLS or OMP  
    VALUE  = *epsilon* for IRLS and *spasity* for OMP  
    DICT_SIZE = 128 or 256  
    _This will start displaying three images, which are: the current selected patch, the proposed edited patch and the state of the modified image at present. This will also store the results in a directory `./images/results/`_

____
