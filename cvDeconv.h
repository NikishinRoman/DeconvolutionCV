/**
 * @file : cvDeconv.h
 * @author : Arnaud Mallen
 * @date : 07/06/2016
 * @version : 1.0
 *
 * @brief : Deconvolution algorithms
 *
 * @section DESCRIPTION
 *
 *
 *
 */

#ifndef DEBLURRINGCV_OPENCVDECONV_H
#define DEBLURRINGCV_OPENCVDECONV_H

#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cvTools.h"

/*
 * TODO : implement accelerated versions of all Richardson-Lucy algorithms
 * "Acceleration of iterative image resoration algorithms, Biggs & Andrews, 1997
 */
class cvDeconv
{
public:
    static void pdeDeconv();

    /*
     * Wiener filter deconvolution, with input periodization
     *  - Adds full borders (half-size of the image in every directions)
     *  - Transforms input kernel in frequency space
     *  - Applies the corresponding wiener filter to the image
     *  - Returns the filtered input image, same size as input.
     *
     * @param imbn : Input blurred noisy image
     * @param deconv : in-out deconvolved image. no need to allocate
     * @param kernel : blurring kernel
     * @param mu : regularization parameter
     */
    static void wienerDeconv(const cv::Mat & imbn, cv::Mat & deconv, const cv::Mat & kernel, double mu);

    /*
     * Richardson-lucy deconvolution
     * TODO : Use correlation for the second computation (OK for now on symmetric kernels)
     *
     * @param
     */
    static void richardsonLucyDeconv(const cv::Mat & imbn, cv::Mat & deconv, const cv::Mat & kernel, int numit );

    /*
     * Richardson-lucy deconvolution
     * TODO : Use correlation for the second computation (OK for now on symmetric kernels)
     *
     * @param
     */
    static void richardsonLucyDeconvTikh(const cv::Mat &imbn, cv::Mat &deconv, const cv::Mat &kernel, double mu,
                                         int numit);
    static void fistaDeconv();
    static void fastPdeDeconv();
    static void fastRichardsonLucyDeconv();
};


void cvDeconv::richardsonLucyDeconv(const cv::Mat & imbn, cv::Mat & deconv, const cv::Mat & kernel, int numit)
{
    // Init output as the input image
    deconv = imbn.clone();

    // Carry out the iterations
    for (int it = 0; it < numit; it++)
    {
        // Temporary matrix
        cv::Mat ratio;
        cv::filter2D(deconv, ratio, deconv.depth(), kernel,cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

        cv::divide(imbn, ratio, ratio);

        cv::filter2D(ratio, ratio, ratio.depth(), kernel,cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

        // Apply iteration on the estimate
        cv::multiply(deconv,ratio,deconv);
    }
}

void cvDeconv::richardsonLucyDeconvTikh(const cv::Mat &imbn, cv::Mat &deconv, const cv::Mat &kernel, double mu,
                                        int numit)
{
    // Init output as the input image
    deconv = imbn.clone();

    // Carry out the iterations
    for (int it = 0; it < numit; it++)
    {
        // Temporary matrix
        cv::Mat ratio;
        cv::filter2D(deconv, ratio, deconv.depth(), kernel,cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

        cv::divide(imbn, ratio, ratio);

        cv::filter2D(ratio, ratio, ratio.depth(), kernel,cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

        // TV Regularization
        cv::Mat denom ;
        cv::Laplacian(deconv,denom,deconv.depth(),1,1,0,cv::BORDER_REFLECT);
        denom = 1.0 - 2.0 * mu * denom;
        /*
        std::cout<<denom.size()<<std::endl;
        std::cout<<cvTools::max(denom)<<std::endl;
        std::cout<<cvTools::min(denom)<<std::endl;
        cvTools::displayImage(denom,"DENOM");
         */
        cv::divide(ratio,denom,ratio);

        // Apply iteration on the estimate
        cv::multiply(deconv,ratio,deconv);

    }
}

void cvDeconv::wienerDeconv(const cv::Mat& imbn, cv::Mat & deconv, const cv::Mat & kernel, double mu)
{
    double minDenom = std::sqrt(std::numeric_limits<double>::epsilon());
    // Minimize border effects : size = 2 * size with mirror constraints to obtain periodic image
    cv::copyMakeBorder(imbn,deconv,imbn.rows/2,imbn.rows/2,imbn.cols/2,imbn.cols/2,cv::BORDER_REFLECT);

    // Transform image to frequency space
    cv::dft(deconv, deconv, cv::DFT_COMPLEX_OUTPUT);

    // transform kernel to frequency space (Optical Transfer Fuction)
    cv::Mat otf;
    cvTools::psf2otf(kernel, otf, deconv.size());

    // Actual filtering, with regularization if frequencies amplitude is too small
    std::complex<double> * otf_pnt = (std::complex<double> *) otf.ptr();
    std::complex<double> * tmp_pnt = (std::complex<double> *) deconv.ptr();
    for (int i = 0; i < otf.rows * otf.cols; i++)
    {
        std::complex<double> conjO(otf_pnt[i].real(), -otf_pnt[i].imag());
        std::complex<double> denom = conjO * otf_pnt[i] + mu;
        if (std::abs(denom) < minDenom )
        {
            tmp_pnt[i] *= conjO / minDenom;
        }else
        {
            tmp_pnt[i] *= conjO / denom;
        }
    }

    // Back in image space
    cv::dft(deconv, deconv, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);

    // Crop useless data
    cv::Rect outROI(imbn.rows/2-kernel.rows/2, imbn.cols/2-kernel.cols/2, imbn.rows, imbn.cols);
    deconv = deconv(outROI);
}

#endif //DEBLURRINGCV_OPENCVDECONV_H
