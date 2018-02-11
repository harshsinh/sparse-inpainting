#include "../include/omp.h"

#include <vector>

cv::Mat_<double> omp (const cv::Mat_<double>& Dict, const cv::Mat_<double>& X, const double sparcity)
{

    cv::Mat_<double> phi;
    cv::Mat_<double> r = X.clone();
    cv::Mat_<double> a;

    int t = 0;
    const int N = Dict.cols;

    std::vector<int> columns;

    while (t != sparcity)
    {
        double maxproduct = 0;
        int maxid = 0;

        for (int i = 0; i < N; ++i)
        {
            double product = r.dot (Dict.col(i));
            if (product >= maxproduct) {
                maxproduct = product;
                maxid = i;
            }
        }

        phi.col(t) = Dict.col(maxid);
        columns.push_back (maxid);

        a = (phi.t().mul(phi)).inv(cv::DECOMP_SVD).mul(phi.t()).mul(X);

        r = X - phi.mul(a);

        ++t;
    }

    /* Making the sparse representation vector from a */
    cv::Mat_<double> sparse_rep = cv::Mat_<double>::zeros (Dict.cols, 1);

    for (std::vector<int>::iterator it = columns.begin(); it != columns.end(); ++it)
    {
        int i = *it;
        sparse_rep.at<double>(i, 1) = a.at<double>(i, 1);
    }

    return sparse_rep;
}