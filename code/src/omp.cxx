#include "../include/omp.h"

#include <vector>
#include <iostream>
#include <cmath>

cv::Mat_<double> omp (const cv::Mat_<double>& Dict, const cv::Mat_<double>& X, const double sparcity)
{

    cv::Mat_<double> phi;
    cv::Mat_<double> a;
    cv::Mat r = X.clone();
    r.convertTo (r, CV_64FC1);

    int t = 0;
    const int N = Dict.cols;

    std::vector<int> columns;

    while (t != sparcity)
    {
        double maxproduct = 0;
        int maxid = 0;

        for (int i = 0; i < N; ++i)
        {

            cv::Mat col = Dict.col(i);
            col.convertTo (col, CV_64FC1);

            double product = std::abs (r.dot (col));
            if (product >= maxproduct) {
                maxproduct = product;
                maxid = i;
            }
        }

        if (phi.empty())
            phi = Dict.col(maxid);
        else
            cv::hconcat (phi, Dict.col(maxid), phi);

        columns.push_back (maxid);

        auto phi_ = (phi.t() * phi).inv(cv::DECOMP_SVD);

        a = phi_ * phi.t();
        a = a * X;

        r = X - (phi * a);

        ++t;
    }

    /* Making the sparse representation vector from a */
    cv::Mat_<double> sparse_rep = cv::Mat_<double>::zeros (Dict.cols, 1);

    int count = 0;
    for (const auto& i : columns)
        sparse_rep.at<double>(i, 1) = a.at<double> (count++, 1);

    return sparse_rep;
}