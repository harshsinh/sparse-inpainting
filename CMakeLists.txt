cmake_minimum_required (VERSION 2.8)
project (inpainting)
find_package (OpenCV REQUIRED)
add_compile_options (-std=c++11)

add_executable (create_mask code/src/create_mask.cxx )
target_link_libraries (create_mask ${OpenCV_LIBS} )

add_executable (sparse_inpaint code/src/sparse_inpaint.cxx)
target_link_libraries (sparse_inpaint ${OpenCV_LIBS})

add_executable (test code/src/test.cxx code/src/omp.cxx code/src/irls.cxx)
target_link_libraries (test ${OpenCV_LIBS})
