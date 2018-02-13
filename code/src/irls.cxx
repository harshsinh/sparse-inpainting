#include "../include/irls.h"
#include <iostream>

cv::Mat updateW (const cv::Mat_<double>& a)
{

    cv::Mat w;
    cv::pow (a, 10, w);

    w = cv::Mat::diag(w);

    return w;
}

cv::Mat_<double> irls (const cv::Mat_<double>& D, const cv::Mat_<double>& X, const double epsilon)
{

    cv::Mat_<double> r = X.clone();
    cv::Mat_<double> a = cv::Mat_<double>::ones (D.cols, 1);
    cv::Mat_<double> w = cv::Mat_<double>::eye (D.cols, D.cols);
    
    r.convertTo (r, CV_64FC1);

    w = updateW (a);

    double error = cv::norm(r);

    int count  = 0;
    while (error > epsilon)
    {
        auto minor_product = (D * (w * w) * D.t());
        a = (w * w) * D.t() * (minor_product.inv(cv::DECOMP_SVD)) * X;

        w = updateW (a);

        r = X - D*a;

        error = cv::norm(r);
        std::cout << "Error: " << error << std::endl;

        count++;
        std::cout << "Iteration : " << count << std::endl;
    }

    return a;
}