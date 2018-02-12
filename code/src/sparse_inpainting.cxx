#include "../include/sparse_inpainting.h"

#include "../include/omp.h"
#include "../include/irls.h"

#include <string>
#include <math.h>
#include <vector>
#include <utility>
#include <iostream>

cv::Mat patch (const cv::Point& p, const cv::Mat& image, int size)
{

    cv::Mat im;
    im  = image (cv::Range (p.x - (size - 1)/2, p.x + (size + 2)/2),
                 cv::Range (p.y - (size - 1)/2, p.y + (size + 2)/2));

    return im;
}

cv::Mat normalizeVec (const cv::Mat_<double>& r)
{

    auto vec = r;
    vec.convertTo (vec, CV_64FC1);
    auto norm = cv::norm (vec);

    vec = vec/norm;

    vec.convertTo (vec, r.type());

    return vec;
}

cv::Mat_<double> normalizeDict (const cv::Mat_<double>& dict)
{

    cv::Mat_<double> normed_D;
    
    for (int i = 0; i < dict.cols; ++i)
    {
        auto col_ = dict.col(i);
        col_.convertTo (col_, CV_64FC1);
        auto norm = cv::norm (col_);
        col_ = col_/norm;

        if (normed_D.empty())
            normed_D = col_;
        else
            cv::hconcat (normed_D, col_, normed_D); 
    }

    return normed_D;
}

cv::Mat removeRows (const cv::Mat& mat, const cv::Mat_<bool>& rows_)
{

    cv::Mat removed_mat;

    for (int i = 0; i < mat.rows; ++i)
    {
        if (rows_.at<bool> (i)) continue;
        if (removed_mat.empty())
            removed_mat = mat.row (i);
        else
            cv::vconcat (removed_mat, mat.row(i), removed_mat);
    }

    return removed_mat;
}

void getBoundary (std::map<std::pair<int, int>, double>& priorities,
                  cv::Mat& M)
{

    priorities.clear();

    for (int i = 3; i < M.rows - 4; ++i)
    {
        for (int j = 3; j < M.cols - 4; ++j)
        {
            cv::Mat region = patch(cv::Point(i, j), M, 3);
            double sum = cv::sum(region)[0];

            if (sum < 255*9 && sum > 0) {
                
                region = patch (cv::Point(i, j), M, 8);
                double sum = cv::sum(region)[0];
                
                priorities.emplace (std::pair<int, int> (i, j), 1 / sum);
            }
        }
    }
}

void updateBoundary (std::map<std::pair<int, int>, double>& priorities,
                     cv::Mat& M, cv::Point Last, int LastID)
{
    for (int i = Last.x - 5; i <= Last.x + 5; ++i)
    {
        for (int j = Last.y - 5; j <= Last.y + 5; ++j)
        {
            auto it = priorities.find(std::pair<int, int> (i, j));
            if (it != priorities.end())
                priorities.erase (it);

            cv::Mat region = patch (cv::Point(i, j), M, 3);
            double sum = cv::sum(region)[0];

            if (sum < 255*9 && sum > 0) {

                
                region = patch (cv::Point(i, j), M, 8);
                double sum = cv::sum(region)[0];

                priorities.emplace (std::pair<int, int>(i, j), 1/sum);
            }
        }
    }
}

cv::Mat_<double> sparseInpaint (const cv::Mat_<double>& Image,
                                const cv::Mat_<double>& Mask,
                                const cv::Mat_<double>& Dictionary,
                                const double Value, const std::string& Method)
{

    cv::Mat_<double> I = Image.clone() / 255;
    cv::Mat M;
    Mask.convertTo (M, CV_8UC1); 
    cv::Mat_<double> D = Dictionary.clone();

    auto D_ = normalizeDict (D);

    std::map<std::pair<int, int>, double> priorities;
    getBoundary (priorities, M);

    while (priorities.size())
    {
        /*Choose patch with max priority*/
        auto it = std::max_element (priorities.begin(), priorities.end(),
                                 [](const std::pair<std::pair<int, int>, double>& x,
                                    const std::pair<std::pair<int, int>, double>& y)
                                    { return x.second < y.second; });
        const cv::Point center (it->first.first, it->first.second);

        /*Reshape patch into a vector*/
        auto X = patch (center, I, 8);
        auto X_ = X.clone();
        X_ = X_.reshape(0, X.rows*X.cols);
        
        auto maskP = patch (center, M, 8);
        auto maskP_ = maskP.clone();
        maskP_ = maskP_.reshape(0, maskP.rows * maskP.cols);

        /*Remove the rows from the vector that have the error bits*/
        auto X_reduced = removeRows (X_, maskP);
        const auto& D_reduced = removeRows (D_, maskP);

        /*Reconstruct using the original dictionary*/
        cv::Mat_<double> a;
        if (Method == "IRLS")
            a = irls (D_reduced, X_reduced, Value);
        else
            a = omp (D_reduced, X_reduced, Value);

        cv::Mat R = D_ * a;

        // R = R * 255;
        R.convertTo (R, X.type());
        R = R.reshape (0, X.rows);

        /*Magnifying patches by 10 times to show*/
        cv::Mat Xsz, Rsz, maskPsz;

        cv::resize (X, Xsz, cv::Size (160, 160), 0, 0, cv::INTER_NEAREST);
        cv::resize (R, Rsz, cv::Size (160, 160), 0, 0, cv::INTER_NEAREST);
        cv::resize (maskP, maskPsz, cv::Size (160, 160), 0, 0, cv::INTER_NEAREST);

        cv::namedWindow ("image", cv::WINDOW_NORMAL);
        cv::imshow ("image", I);

        // cv::imshow ("Complete Mask", M);        
        // cv::imshow ("Mask", maskPsz);
        cv::imshow ("Selected Patch", Xsz);
        cv::imshow ("Proposed Patch", Rsz);

        R.copyTo (X, maskP);
        
        cv::resize (X, Xsz, cv::Size (160, 160), 0, 0, cv::INTER_NEAREST);
        // cv::namedWindow ("Modified Patch", cv::WINDOW_NORMAL);
        // cv::imshow ("Modified Patch", Xsz);

        cv::waitKey (1);

        /*Remove this patch from mask*/
        maskP.setTo (0);

        /*Update the boundary and priority*/
        updateBoundary (priorities, M, center, 0);

        // cv::Mat mod = I (cv::Range(center.x - 3, center.x + 4),
        //                  cv::Range(center.y - 3, center.y + 4));
        // X.copyTo (mod);

        // cv::imshow ("image", I);
    }

    return I;
}