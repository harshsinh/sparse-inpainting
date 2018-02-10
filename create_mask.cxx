#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src,cloneimg;
bool mousedown;
vector<vector<Point> > contours;
vector<Point> pts;

void onMouse( int event, int x, int y, int flags, void* userdata )
{
    Mat img = *((Mat *)userdata);

    if( event == EVENT_LBUTTONDOWN )
    {
        mousedown = true;
        contours.clear();
        pts.clear();
    }

    if( event == EVENT_LBUTTONUP )
    {
        mousedown = false;
        if(pts.size() > 2 )
        {
            Mat mask(img.size(),CV_8UC1);
            mask = 0;
            contours.push_back(pts);
            drawContours(mask,contours,0,Scalar(255),-1);
            Mat masked(img.size(),CV_8UC3,Scalar(255,255,255));
            src.copyTo(masked,mask);
            src.copyTo(cloneimg);
            namedWindow("Masked Image", WINDOW_NORMAL);
            imshow ("Masked Image", masked);
            imwrite ("mask.jpg", masked);
        }
    }

    if(mousedown)
    {
        if(pts.size() > 2 )
            line(img,Point(x,y),pts[pts.size()-1],Scalar(0,255,0));

        pts.push_back(Point(x,y));

        imshow("Create Mask", img);
    }
}

int main (int argc, const char** argv)
{
    /* image is taken as argument */
    src = imread(argv[1]);
    if(src.empty())
    {
        std::cout << "INVALID Image" << std::endl;
        return -1;
    }

    namedWindow("Create Mask", WINDOW_NORMAL);
    cloneimg = src.clone();
    setMouseCallback("Create Mask", onMouse, &cloneimg);
    imshow("Create Mask", src);

    waitKey(0);
    return 0;
}