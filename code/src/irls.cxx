#include "../include/irls.h"

cv::Mat_<double> irls (const cv::Mat_<double>& Dict, const cv::Mat_<double>& X, const double epsilon)
{

    cv::Mat_<double> r = X.clone();
    cv::Mat_<double> a = cv::Mat_<double>::ones (r.size());
    cv::Mat_<double> w = cv::Mat_<double>::eye (r.rows, r.rows) * a;
    
    
    w = w * a;

    cv::Mat_<double> norm = (r.t().mul(r));
    double error = norm.at<double>(0, 0);

    while (error > epsilon)
    {
        auto minor_product = (Dict * w) * (w * Dict.t());
        a = w.mul(w) * Dict.t() * (minor_product.inv(cv::DECOMP_SVD)) * X;

        w = cv::Mat_<double>::eye (X.rows, X.rows) * a;
        w = w * a;

        cv::Mat_<double> norm = (r.t().mul(r));
        double error = norm.at<double>(0, 0);
    }

    return a;
}