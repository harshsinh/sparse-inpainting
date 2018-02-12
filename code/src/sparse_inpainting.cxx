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

cv::Mat removeRows (const cv::Mat_<double>& mat, const cv::Mat_<bool>& rows_)
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

void getBoundary (std::vector<cv::Point>& B,
                  std::vector<double>& priority,
                  cv::Mat_<double>& M)
{

    B.clear();
    priority.clear();

    for (int i = 3; i < M.rows - 4; ++i)
    {
        for (int j = 3; j < M.cols - 4; ++j)
        {
            cv::Mat region = patch(cv::Point(i, j), M, 3);
            double sum = cv::sum(region)[0];

            if (sum < 255*9 && sum > 0) {

                std::cout << "sum of region " << cv::Point(i, j) << ": " << sum << std::endl;
                B.push_back (cv::Point(i, j));
                region = patch (cv::Point(i, j), M, 8);
                double sum = cv::sum(region)[0];
                priority.push_back (1/sum);

            }

        }
    }
    return;
}

void updateBoundary (std::vector<cv::Point>& B, std::vector<double>& p_,
                     cv::Mat& M, cv::Point Last, int LastID)
{

    auto start = B.begin();
    auto end   = B.end();
    auto p_begin = p_.begin();

    for (int i = Last.x - 5; i <= Last.x + 5; ++i)
    {

        for (int j = Last.y - 5; j <= Last.y + 5; ++j)
        {

            auto f = std::find (start, end, cv::Point(i, j));

            if (f != end) {

                B.erase (f);
                p_.erase ((f - start) + p_begin);

            }

            std::cout << "Point: " << cv::Point(i, j) << std::endl;

            cv::Mat region = patch (cv::Point(i, j), M, 3);
            double sum = cv::sum(region)[0];

            if (sum < 255*64 && sum > 255*16) {

                B.push_back (cv::Point(i, j));
                region = patch (cv::Point(i, j), M, 8);
                double sum = cv::sum(region)[0];

                p_.push_back (1/sum);
            }
        }
    }
}

cv::Mat_<double> sparseInpaint (const cv::Mat_<double>& Image,
                                const cv::Mat_<double>& Mask,
                                const cv::Mat_<double>& Dictionary,
                                const double Value, const std::string& Method)
{

    cv::Mat_<double> I = Image.clone();
    cv::Mat_<double> M = Mask.clone();
    cv::Mat_<double> D = Dictionary.clone();

    auto D_ = normalizeDict (D);

    std::vector<cv::Point> B;
    std::vector<double> priority;
    getBoundary (B, priority, M);

    std::cout << "sz: " << priority.size() << std::endl;

    while (priority.size() != 0)
    {
        /*Choose patch with max priority*/
        auto it = std::max_element (priority.begin(), priority.end()) - priority.begin();
        cv::Point center = B[it];

        /*Reshape patch into a vector*/
        auto X = patch (center, I, 8);
        auto X_ = X.clone();
        X_ = X_.reshape(0, X.rows*X.cols);
        
        auto maskP = patch (center, M, 8);
        maskP = maskP.clone();
        auto maskP_ = maskP.reshape(0, maskP.rows * maskP.cols);

        /*Remove this patch from mask*/
        cv::Mat aux = M(cv::Range(center.x - 3, center.x + 4),
                         cv::Range(center.y - 3, center.y + 4));// = cv::Mat_<double>::zeros(8, 8);
        cv::Mat zeros = cv::Mat_<double>::zeros(8, 8);
        zeros.copyTo (aux);

        /*Update the boundary and priority*/
        updateBoundary (B, priority, M, center, it);

        /*Remove the rows from the vector that have the error bits*/
        std::cout << "X_ size, D size: " << X_.size() << ", " << D.size() << std::endl;
        auto X_reduced = removeRows (X_, maskP);
        const auto& D_reduced = removeRows (D, maskP);

        /*Normalize all the involved vectors*/
        X_ = X_/255;

        auto X_reduced_ = X_reduced/255;
        auto D_reduced_ = normalizeDict (D_reduced);

        /*Reconstruct using the original dictionary*/
        cv::Mat_<double> a;
        if (Method == "IRLS")
            a = irls (D_reduced_, X_reduced_, Value);
        else
            a = omp (D_reduced_, X_reduced_, Value);

        cv::Mat R = D_ * a;

        // R = R * 255;
        R.convertTo (R, X.type());
        R = R.reshape (0, X.rows);

        /*Magnifying patches by 10 times to show*/
        cv::Mat Xsz, Rsz, maskPsz;

        cv::resize (X, Xsz, cv::Size (160, 160), 0, 0, cv::INTER_NEAREST);
        cv::resize (R, Rsz, cv::Size (160, 160), 0, 0, cv::INTER_NEAREST);
        cv::resize (M, maskPsz, cv::Size (160, 160), 0, 0, cv::INTER_NEAREST);

        cv::namedWindow ("image", cv::WINDOW_NORMAL);
        cv::imshow ("image", I/255);
        
        cv::imshow ("Mask", maskPsz);
        cv::imshow ("Selected Patch", Xsz);
        cv::imshow ("Proposed Patch", Rsz);

        cv::waitKey (5);

        std::cout << "R type: " << R.type() << std::endl;

        maskP.convertTo (maskP, CV_8U);

        R.copyTo (X, maskP);

        // cv::Mat mod = I (cv::Range(center.x - 3, center.x + 4),
        //                  cv::Range(center.y - 3, center.y + 4));
        // X.copyTo (mod);

        // cv::imshow ("image", I);
    }

    return I/255;
}