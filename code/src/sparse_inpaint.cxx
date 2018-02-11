#include "../include/sparse_inpaint.h"

#include <iostream>
#include <string>

cv::Mat image, mask;
const std::string topdir = "../";

// int getPriority (cv::Point patch_pos)
// {}

// void getBoundary ()
// {}

int main (int argc, char** argv)
{

    image = cv::imread (topdir + "images/maskedimage.JPG");
    mask  = cv::imread (topdir + "images/mask.jpg");

    cv::namedWindow ("Masked Image", cv::WINDOW_NORMAL);
    cv::imshow ("Masked Image", image);

    cv::waitKey(0);
    return 0;
}