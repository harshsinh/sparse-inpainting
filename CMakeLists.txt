cmake_minimum_required (VERSION 2.8)
project (inpainting)
find_package (OpenCV REQUIRED)

add_executable (create_mask code/src/create_mask.cxx )
target_link_libraries (create_mask ${OpenCV_LIBS} )

add_executable (sparse_inpaint code/src/sparse_inpaint.cxx)
target_link_libraries (sparse_inpaint ${OpenCV_LIBS})

add_executable (omp code/src/omp.cxx)
target_link_libraries (omp ${OpenCV_LIBS})
