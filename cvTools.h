/**
 * @file : cvTools.h
 * @author : Arnaud Mallen
 * @date : 07/06/2016
 * @version : 1.0
 *
 * @brief : OpenCV general tools
 *
 * @section DESCRIPTION
 *
 *
 */

#ifndef DEBLURRINGCV_OPENCVUTILS_H
#define DEBLURRINGCV_OPENCVUTILS_H

#include <string>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "shift.hpp"

/*
 * Provides general purpose tools for image processing
 */
class cvTools{
public:

    /*
     * Loads an image file into a cv::Mat
     *
     * @param imagePath : path to the image to be loaded
     * @return a cv::Mat containing the image bands
     */
    static cv::Mat loadImageToGrayCvMat(std::string imagePath);

    /*
     * Blurs an image with a given blurring kernel with mirror boundary conditions
     *
     * @param image : the image to be blurred
     * @param kernel : the blurring kernel
     */
    static void blurredGrayImage(cv::Mat & image, const cv::Mat & kernel);

    /*
     * Applies a gaussian noise of given variance and zero-mean to an input image
     *
     * @param image : the image to be "noised"
     * @param var : the variance of the gaussian noise
     */
    static void applyGaussianNoise(cv::Mat & image, double var);

    /*
     * Displays an input image in a specific window
     *
     * @param image : the image to be displayed
     * @param windowName : the name of the window
     */
    static void displayImage(const cv::Mat & image, std::string windowName = "Image");

    /*
     * Applies gaussian blur + noise to an image
     *
     * @param image : the image to be degraded
     * @param kernel : the blurring kernel
     * @param sigma : the standard deviation of the gaussian noise
     */
    static void blurNoise(cv::Mat & image, const cv::Mat & kernel, double sigma);

    /*
     * Matlab-like operation on an image. Shifts the pixels values.
     *
     * @param image : the image to be shifted
     * @param shiftedImage : the shifted image
     * @param shift : the 2-D parameters of the shift (can be float-valued)
     *
     */
    static void circshift(const cv::Mat & image, cv::Mat & shiftedImage, cv::Point2f shift);

    /*
     * Matlab-like operation on an image. Pads the image with given borders and values
     *
     * @param image : the image to be padded
     * @param paddedImage : padded output image
     * @param top : top border size
     * @param bottom : bottom border size
     * @param left : left border size
     * @param right : right border size
     * @param borderType : type of border to be used (cv::BORDER_* available)
     * @param value : value to use in the border (for instance when borderType = BORDER_CONSTANT)
     */
    static void padarray(const cv::Mat & image,
                         cv::Mat & paddedImage,
                         int top, int bottom,
                         int left, int right,
                         int borderType,
                         double value);

    /*
     * Converts a blurring kernel (psf matrix) to an Optical Transfer Function (psf in frequency domain)
     *  - Pads the psf with zeros to obtain desired OTF size
     *  - Applies fourier transform to the padded psf
     *
     *  @param psf : input blurring kernel
     *  @param otf : output optical transfer function
     *  @param s : size of the desired otf
     */
    static void psf2otf(const cv::Mat & psf, cv::Mat & otf, const cv::Size & s);

    /*
     * Obtain a 2-D gaussian blurring kernel of given standard deviation and size
     *
     * @param kernel : output blurring kernel
     * @param size : size of the blurring kernel
     * @param sigma : spatial standard deviation of the kernel
     */
    static void getGaussianKernel(cv::Mat & kernel, int size, double sigma);

    /*
     * Obtain a string describing the data type corresponding to a cv::Mat type
     *
     * @param number : the number given by cv::Mat::getType()
     * @return std::string describing OpenCV datatype
     */
    static std::string getImageType(int number);

    /*
     * Computes the maximum value of the first band of a cv::Mat
     *
     * @param m : input cv::Mat
     * @return double maximum value
     */
    static double max(const cv::Mat & m);

    /*
     * Computes the standard deviation of the first band of a cv::Mat
     *
     * @param m : input cv::Mat
     * @return double standard deviation
     */
    static double std(const cv::Mat & m);
};

double cvTools::std(const cv::Mat & m)
{
    cv::Scalar std, mean;
    cv::meanStdDev(m,mean,std);
    return std[0];
}

double cvTools::max(const cv::Mat & m)
{
    double min, max;
    cv::minMaxLoc(m, &min, &max);
    return max;
}

std::string cvTools::getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

void cvTools::getGaussianKernel(cv::Mat & kernel, int size, double sigma)
{
    cv::Mat kernelX = cv::getGaussianKernel(size,sigma);
    cv::Mat kernelY = cv::getGaussianKernel(size,sigma);
    kernel = kernelX * kernelY.t();
}

void cvTools::padarray(const cv::Mat & image,cv::Mat & paddedImage, int top, int bottom, int left, int right, int borderType, double value)
{
    cv::copyMakeBorder(image, paddedImage, top, bottom, left, right, borderType, value);
}

void cvTools::circshift(const cv::Mat &image, cv::Mat &shiftedImage, cv::Point2f shiftCoords)
{
    shift(image, shiftedImage, shiftCoords, cv::BORDER_REFLECT);
}

void cvTools::psf2otf(const cv::Mat & psf, cv::Mat & otf, const cv::Size & s)
{
    padarray(psf, otf, 0, s.height - psf.rows, 0, s.width - psf.cols, cv::BORDER_CONSTANT, 0);
    cv::dft(otf,otf,cv::DFT_COMPLEX_OUTPUT);
}
cv::Mat cvTools::loadImageToGrayCvMat(std::string imagePath)
{
    return cv::imread(imagePath.c_str(), CV_LOAD_IMAGE_ANYDEPTH);
}

void cvTools::blurNoise(cv::Mat & image, const cv::Mat & kernel, double sigma)
{
    blurredGrayImage(image,kernel);
    applyGaussianNoise(image,sigma);
}

void cvTools::applyGaussianNoise(cv::Mat & image, double sigma)
{
    cv::Mat noise = cv::Mat(image.size(),CV_64F);
    cv::randn(noise, 0, sigma);
    image = image + noise;
}

void cvTools::blurredGrayImage(cv::Mat & image,const cv::Mat & kernel)
{
    cv::filter2D(image, image, image.depth(), kernel,cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
}

void cvTools::displayImage(const cv::Mat &image, std::string windowName)
{
    namedWindow( windowName, cv::WINDOW_AUTOSIZE );
    imshow( windowName, image);
    cv::waitKey(0);
}

#endif //DEBLURRINGCV_OPENCVUTILS_H
