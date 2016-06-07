#include <iostream>
#include <chrono>

#include "cvTools.h"
#include "cvDeconv.h"
#include "cvQM.h"

using namespace std::chrono;

/*
 * Takes one argument : the path of an image to distort & deconvolve
 */
int main(int argc, char * argv[]) {
    // TODO : Input parsing
    std::cout << "Loading "<<argv[1] << std::endl;
    cv::Mat image = cvTools::loadImageToGrayCvMat(argv[1]);
    cv::Mat ref = cvTools::loadImageToGrayCvMat(argv[1]);
    cv::normalize(image, image, 0.0, 1.0, CV_MINMAX, CV_64F);
    cv::normalize(ref, ref, 0.0, 1.0, CV_MINMAX, CV_64F);

    // Blur and add noise to input
    cv::Mat kernel;
    cvTools::getGaussianKernel(kernel, 21, 3);
    double sigma=0.01;
    cvTools::blurNoise(image, kernel, sigma);

    cv::Mat deconv;
    high_resolution_clock ::time_point t1 = high_resolution_clock::now();
    cvDeconv::wienerDeconv(image, deconv, kernel, sigma*sigma/std::pow(cvTools::std(image),2));
    high_resolution_clock ::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(t2-t1).count();

    std::cout << "Wiener deconvolution : "<<(double)duration / (double)1000000000 <<" s "<<std::endl;

    std::cout<<"PSNR image : "<<cvQM::psnr(ref, image, cvTools::max(ref))<<std::endl;
    std::cout<<"PSNR deconv : "<<cvQM::psnr(ref, deconv, cvTools::max(ref))<<std::endl;
    std::cout<<"SSIM image : "<<cvQM::ssim(ref, image)<<std::endl;
    std::cout<<"SSIM deconv : "<<cvQM::ssim(ref, deconv)<<std::endl;

    cvTools::displayImage(image,"Degraded Image");
    cvTools::displayImage(deconv,"Deconv Image");
    cvTools::displayImage(ref,"Ref Image");


    return 0;
}