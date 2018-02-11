#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

cv::Mat_<double> omp (const cv::Mat_<double>& Dict, const cv::Mat_<double>& X, const double sparcity);