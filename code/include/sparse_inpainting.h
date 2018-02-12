#pragma once

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

#include <map>

cv::Mat patch (const cv::Point& p, const cv::Mat& image, int size);

cv::Mat normalizeVec (const cv::Mat_<double>& r);

cv::Mat_<double> normalizeDict (const cv::Mat_<double>& dict);

cv::Mat removeRows (const cv::Mat& mat, const cv::Mat_<bool>& rows_);

void getBoundary (std::map<std::pair<int, int>, double>& priorities,
                  cv::Mat& M);

void updateBoundary (std::map<std::pair<int, int>, double>& priorities,
                     cv::Mat& M, cv::Point Last, int LastID);

cv::Mat_<double> sparseInpaint (const cv::Mat_<double>& Image,
                                const cv::Mat_<double>& Mask,
                                const cv::Mat_<double>& Dictionary,
                                const double Value,
                                const std::string& Method);