#include "StudentPreProcessing.h"
#include "HereBeDragons.h"
#include "ImageFactory.h"
#include "ImageIO.h"
#include <array>
#include <iostream>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>

IntensityImage * StudentPreProcessing::stepToIntensityImage(const RGBImage &image) const {
	return nullptr;
}

IntensityImage * StudentPreProcessing::stepScaleImage(const IntensityImage &image) const {
	return nullptr;
}

cv::Mat hypo(const cv::Mat& sobel_x, const cv::Mat& sobel_y){ //, const cv::Mat& sobel_d1, const cv::Mat& sobel_d2) {
    assert(sobel_x.rows == sobel_y.rows);
    assert(sobel_x.cols == sobel_y.cols);

    cv::Mat magnitude;
    magnitude.create(sobel_x.rows, sobel_x.cols, CV_8UC1);
   
    for (int x = 0; x < sobel_x.rows; x++) {
        for (int y = 0; y < sobel_x.cols; y++) {
            uint32_t x_val = sobel_x.at<uchar>(x, y);
            uint32_t y_val = sobel_y.at<uchar>(x, y);
           /* uint32_t d1_val = sobel_d1.at<uchar>(x, y);
            uint32_t d2_val = sobel_d2.at<uchar>(x, y);*/
            magnitude.at<uchar>(x, y) = static_cast<uchar>(sqrt(pow(x_val, 2) + pow(y_val, 2))); // +pow(d1_val, 2) + pow(d2_val, 2)));
        }
    }

    return magnitude;
}


cv::Mat constructGaussianFilter(const double& sigma, const unsigned int size = 2) {
    cv::Mat gaussian_filter = (cv::Mat_<double>(5, 5));

    double r, s = 2.0 * sigma * sigma;  // Assigning standard deviation to 1.0
    double sum = 0.0;                   // Initialization of sun for normalization
    for (int x = -2; x <= 2; x++)       // Loop to generate 5x5 kernel
    {
        for (int y = -2; y <= 2; y++)
        {
            r = sqrt(x * x + y * y);
            gaussian_filter.at<double>(x + 2,y + 2) = (exp(-(r * r) / s)) / ((atan(1)*4)* s);
            sum += gaussian_filter.at<double>(x + 2,y + 2);
        }
    }

    for (int x=0; x < 5; x++) {
        for (int y=0; y < 5; y++) {
            gaussian_filter.at<double>(x,y) /= sum;
        }
    }

    return gaussian_filter;
}

IntensityImage* applyGaussianBlur(const IntensityImage& image, const double &sigma) {
#define SAVE_IMAGE 1
    // Make a matrix for the 'old' image
    cv::Mat unedited_image_matrix;

    // Convert intensity image to values for the matrix, this is done by reference.
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(image, unedited_image_matrix);
   
    // Construct The Gaussian filter
    auto gaussian_filter = constructGaussianFilter(sigma);
    
#ifdef DEBUG
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            std::cout << gaussian_filter.at<double>(i, j) << "  ";
        }
        std::cout << '\n';
    }
#endif // DEBUG

    // Make new matrix for edited image ( w/ Gaussian blur)
    cv::Mat edited_image_matrix;

    // Do the matrix calculations
    filter2D(unedited_image_matrix, edited_image_matrix, CV_8U, gaussian_filter, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // Make a new intensity image for the new converted image
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();

    // Convert new matrix to new intensityimage
    HereBeDragons::NoWantOfConscienceHoldItThatICall(edited_image_matrix, *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("gaussian_filter_student.png"));
#endif // SAVE_IMAGE

    // return the new intensityimage
    return edited_intensity_image;
}

IntensityImage* applySobel(const IntensityImage& image) {
#define SAVE_IMAGE 1
#define DEBUG 1
    // Make a matrix for the 'old' image
    cv::Mat unedited_image_matrix;

    // Convert intensity image to values for the matrix, this is done by reference.
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(image, unedited_image_matrix);

    cv::Mat sobel_x = (cv::Mat_<double>(3, 3) <<
         1, 0, -1,
         2, 0, -2,
         1, 0, -1);
    cv::Mat sobel_y = (cv::Mat_<double>(3, 3) <<
         1,  2,  1,
         0,  0,  0,
        -1, -2, -1);
    cv::Mat sobel_d1 = (cv::Mat_<double>(3, 3) <<
         0,  1,  2,
        -1,  0,  1,
        -2, -1,  0);
    cv::Mat sobel_d2 = (cv::Mat_<double>(3, 3) <<
        -2, -1,  0,
        -1,  0,  1,
         0,  1,  2);

    // Make a matrix for the 'new' image
    cv::Mat edited_image_matrix_x;
    cv::Mat edited_image_matrix_y;
    cv::Mat edited_image_matrix_d1;
    cv::Mat edited_image_matrix_d2;

    

    // do the matrix calculations
    filter2D(unedited_image_matrix, edited_image_matrix_x, CV_8U, sobel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(unedited_image_matrix, edited_image_matrix_y, CV_8U, sobel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(unedited_image_matrix, edited_image_matrix_d1, CV_8U, sobel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(unedited_image_matrix, edited_image_matrix_d2, CV_8U, sobel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    /*cv::pow(edited_image_matrix_x, 2, edited_image_matrix_x);
    cv::pow(edited_image_matrix_y, 2, edited_image_matrix_y);*/
   /* cv::pow(edited_image_matrix_d1, 2, edited_image_matrix_d1);
    cv::pow(edited_image_matrix_d2, 2, edited_image_matrix_d2);*/
    
    //edited_image_matrix_x += edited_image_matrix_y;// +edited_image_matrix_d1 + edited_image_matrix_d2;
    //edited_image_matrix_x /= 2;
    // make a new intensity image

    auto final = hypo(edited_image_matrix_x, edited_image_matrix_y);
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();


    // convert new matrix to new intensityimages
    HereBeDragons::NoWantOfConscienceHoldItThatICall(final , *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("sobell_filter_student4.png"));
#endif // SAVE_IMAGE


    // return the new intensityimage
    return edited_intensity_image;
}



IntensityImage* applyEdgeThinning(const IntensityImage& image) {
#define SAVE_
    // Make a matrix for the 'old' image
    cv::Mat unedited_image_matrix;

    // Convert intensity image to values for the matrix, this is done by reference.
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(image, unedited_image_matrix);
    
    cv::Mat edited_image_matrix;

    edited_image_matrix.create(unedited_image_matrix.rows, unedited_image_matrix.cols, CV_8UC1);
   
    for (int x = 1; x < unedited_image_matrix.rows - 1; x++) {
        for (int y = 1; y < unedited_image_matrix.cols - 1; y++) {
            if (
                unedited_image_matrix.at<uchar>(x - 1, y - 1) == 255 ||
                unedited_image_matrix.at<uchar>(x - 1, y) == 255 ||
                unedited_image_matrix.at<uchar>(x - 1, y + 1) == 255 ||
                unedited_image_matrix.at<uchar>(x, y - 1) == 255 ||
                unedited_image_matrix.at<uchar>(x, y + 1) == 255 ||
                unedited_image_matrix.at<uchar>(x + 1, y - 1) == 255 ||
                unedited_image_matrix.at<uchar>(x + 1, y) == 255 ||
                unedited_image_matrix.at<uchar>(x + 1, y + 1) == 255)
            {
                edited_image_matrix.at<uchar>(x, y) = 255;
            }else {
                edited_image_matrix.at<uchar>(x, y) = 0;
            }
        }
    }

    // Make a new intensity image for the new converted image
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();

    // Convert new matrix to new intensityimage
    HereBeDragons::NoWantOfConscienceHoldItThatICall(edited_image_matrix, *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("edge_thinning.png"));
#endif // SAVE_IMAGE

    // return the new intensityimage
    return edited_intensity_image;

}

/**
* In this function, Canny edge detection will be implemented.
* The main steps of Canny edge are:
*	- Apllying Gaussian blur
*	- Apply Sobell filter
*	- Relate the edgde gradients to directions that can be traced
*	- Tracing valid edges
*	- Hysteresis thresholding to eliminate breaking up of edge contours
*/
IntensityImage* StudentPreProcessing::stepEdgeDetection(const IntensityImage& image) const {
    
    // Apply Gaussian Blurrr
    const double sigma = 0.1;
    auto image_after_gaussian = applyGaussianBlur(image, sigma);

    // Apply Sobell Filter
    auto image_after_sobell = applySobel(*image_after_gaussian);
    
    // Apply Edge Thinning
    auto image_after_edge_thinning = applyEdgeThinning(*image_after_sobell);

    return image_after_edge_thinning;
}

IntensityImage * StudentPreProcessing::stepThresholding(const IntensityImage &image) const {
	return nullptr; 
}