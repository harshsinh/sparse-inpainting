#include "../include/sparse_inpainting.h"
#include "../include/omp.h"
#include "../include/irls.h"

#include <iostream>
#include <string>

const std::string topdir = "../";
const std::string resultsdir = "../images/results/scripts_1000/";

int main (int argc, char** argv)
{

    if (!(argc == 4)) {
        std::cout << "Need 3 arguments: Method Value Dictionary Size"
                  << "\nMethod: IRLS or OMP, Value is Epsilon or Sparcity"
                  << std::endl;
        std::cout << "Number of arguments give: " << argc << std::endl;
        return -1;
    }

    const std::string method = argv[1];
    const double value = std::stod (argv[2]);
    const double D_size = std::stod (argv[3]);

    cv::Mat image = cv::imread (topdir + "images/maskedimage.png", cv::IMREAD_GRAYSCALE);
    cv::Mat mask  = cv::imread (topdir + "images/mask.png", cv::IMREAD_GRAYSCALE);
    cv::Mat dictionary = cv::imread (topdir + "images/dictionary/dictionary.png", cv::IMREAD_GRAYSCALE);

    dictionary = dictionary (cv::Range::all(), cv::Range(0, D_size));

    cv::Mat result = sparseInpaint (image, mask, dictionary, value, method);

    std::string name = resultsdir + method + argv[3] +
                              "_" + argv[2] + ".JPG";


    std::cout << result.size() << std::endl;
    std::cout << "name: " << name << std::endl;
    cv::imwrite (name, result);
    cv::waitKey(0);
    return 0;
}