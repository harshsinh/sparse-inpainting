#pragma once

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"


cv::Mat patch (const cv::Point& p, const cv::Mat& image, int size);

cv::Mat normalizeVec (const cv::Mat_<double>& r);

cv::Mat_<double> normalizeDict (const cv::Mat_<double>& dict);

cv::Mat removeRows (const cv::Mat_<double>& mat, const cv::Mat_<bool>& rows_);

void getBoundary (std::vector<cv::Point>& B, std::vector<double>& priority,
                  cv::Mat_<double>& M);

void updateBoundary (std::vector<cv::Point>& B, std::vector<double>& p_,
                     cv::Mat& M, cv::Point Last, int LastID);

cv::Mat_<double> sparseInpaint (const cv::Mat_<double>& Image,
                                const cv::Mat_<double>& Mask,
                                const cv::Mat_<double>& Dictionary,
                                const double Value,
                                const std::string& Method);