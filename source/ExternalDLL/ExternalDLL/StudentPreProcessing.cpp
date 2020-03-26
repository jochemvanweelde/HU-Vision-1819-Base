#include "StudentPreProcessing.h"
#include "HereBeDragons.h"
#include "ImageFactory.h"
#include "ImageIO.h"
#include <array>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>

IntensityImage * StudentPreProcessing::stepToIntensityImage(const RGBImage &image) const {
	return nullptr;
}

IntensityImage * StudentPreProcessing::stepScaleImage(const IntensityImage &image) const {
	return nullptr;
}

// This function adds up the four sobell filters (x, y, d1 and d2)
// Afther that the highest value within the matrix will be set to a max value 
// of 255. The other values will be scaled approprately to the 255. This is needed
// Because the image depth is 1 byte. ( 8 bits, 255 max)
std::pair<cv::Mat, cv::Mat> hypo(const cv::Mat& sobel_x, const cv::Mat& sobel_y, const cv::Mat& sobel_d1, const cv::Mat& sobel_d2) {
    assert(sobel_x.rows == sobel_y.rows);
    assert(sobel_x.cols == sobel_y.cols);
       
    // create new matrices for the magnitue, return matrix and direction
    cv::Mat magnitude;
    magnitude.create(sobel_x.rows, sobel_x.cols, CV_16U);
    cv::Mat direction;
    direction.create(sobel_x.rows, sobel_x.cols, CV_32FC1);
    cv::Mat return_mat;
    return_mat.create(sobel_x.rows, sobel_x.cols, CV_8U);

   

    uint16_t max = 0;
   
    // Calculate the highest occuring value and calculate the 
    // angle of the edge
    for (int x = 0; x < sobel_x.rows; x++) {
        for (int y = 0; y < sobel_x.cols; y++) {
            uint32_t x_val = sobel_x.at<uchar>(x, y);
            uint32_t y_val = sobel_y.at<uchar>(x, y);
            uint32_t d1_val = sobel_d1.at<uchar>(x, y);
            uint32_t d2_val = sobel_d2.at<uchar>(x, y);
            //magnitude.at<uchar>(x, y) = static_cast<uchar>(sqrt(pow(x_val, 2) + pow(y_val, 2) +pow(d1_val, 2) + pow(d2_val, 2)));
            uint16_t new_val = static_cast<uint16_t>(x_val + y_val + d1_val + d2_val);
            if (new_val > max) { max = new_val; }
            magnitude.at<uint16_t>(x, y) = new_val;
            auto angle = atan2(y, x) * 180.f/ 3.14159265358979323846;
            if (angle < 0.f) { angle += 180; }
            direction.at<float>(x, y) = static_cast<float>(angle);
        }
    }

     // Scale everything down to fit in an 8 bits integer
    for (int x = 0; x < sobel_x.rows; x++) {
        for (int y = 0; y < sobel_x.cols; y++) {
        
            return_mat.at<uchar>(x, y) = static_cast<uchar>( (magnitude.at<uint16_t>(x, y)* UCHAR_MAX) /  max);
        
        }
    }


    return std::pair<cv::Mat, cv::Mat>(return_mat, direction);
}

// This function constructs a 5x5 gaussian filter
// The result is dependant on the sigma provided
cv::Mat constructGaussianFilter(const double& sigma) {
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

// This function constructs and applies Gaussian Blur to an image.
// The amount of blur is determined by the value of 'sigma'
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
            std::cout << gaussian_filter.at<double>(x, y) << "  ";
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
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("gaussian_filter.png"));
#endif // SAVE_IMAGE

    // return the new intensityimage
    return edited_intensity_image;
}


// This function applies the Sobel kernels to an image.
std::pair<IntensityImage*, cv::Mat> applySobel(const IntensityImage& image) {
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
    filter2D(unedited_image_matrix, edited_image_matrix_d1, CV_8U, sobel_d1, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(unedited_image_matrix, edited_image_matrix_d2, CV_8U, sobel_d2, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

#ifdef SAVE_IMAGE
    IntensityImage* x_dir = ImageFactory::newIntensityImage();
    IntensityImage* y_dir = ImageFactory::newIntensityImage();
    // convert new matrix to new intensityimages
    HereBeDragons::NoWantOfConscienceHoldItThatICall(edited_image_matrix_x, *x_dir);
    HereBeDragons::NoWantOfConscienceHoldItThatICall(edited_image_matrix_y, *y_dir);

    ImageIO::saveIntensityImage(*x_dir, ImageIO::getDebugFileName("sobell_x_filter.png"));
    ImageIO::saveIntensityImage(*y_dir, ImageIO::getDebugFileName("sobell_y_filter.png"));

#endif // SAVE_IMAGE


    auto final = hypo(edited_image_matrix_x, edited_image_matrix_y, edited_image_matrix_d1, edited_image_matrix_d2);
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();


    // convert new matrix to new intensityimages
    HereBeDragons::NoWantOfConscienceHoldItThatICall(final.first, *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("sobell_filter.png"));
#endif // SAVE_IMAGE


    // return the new intensityimage
    return std::pair<IntensityImage*, cv::Mat>(edited_intensity_image, final.second);
}


// This function will apply the edge thinning after a sobel filter has gone over the image.
IntensityImage* applyEdgeThinning(const IntensityImage& image, const cv::Mat& direction_matrix) {
#define SAVE_
    // Make a matrix for the 'old' image
    cv::Mat unedited_image_matrix;
    // Convert intensity image to values for the matrix, this is done by reference.
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(image, unedited_image_matrix);
    
    // Create a matrix initialized to 0 of the same size of the original gradient intensity matrix
    cv::Mat edited_image_matrix;
    edited_image_matrix.create(unedited_image_matrix.rows, unedited_image_matrix.cols, CV_8UC1);
   

    for (int x = 1; x < unedited_image_matrix.rows - 1; x++) {
        for (int y = 1; y < unedited_image_matrix.cols - 1; y++) {

            uchar x_dir = 255;
            uchar y_dir = 255;

            // Angle = 0 degrees
            if ((0 <= direction_matrix.at<uchar>(x, y) < 22.5) || (157.5 <= direction_matrix.at<uchar>(x, y) <= 180)) {
                x_dir = unedited_image_matrix.at<uchar>(x, y + 1);
                y_dir = unedited_image_matrix.at<uchar>(x, y - 1);
            } // Angle = 45 degrees
            else if( 22.5 <= direction_matrix.at<uchar>(x, y) < 67.5){
                x_dir = unedited_image_matrix.at<uchar>(x + 1, y - 1);
                y_dir = unedited_image_matrix.at<uchar>(x - 1, y + 1);
            } // Angle = 90 degrees
            else if (67.5 <= direction_matrix.at<uchar>(x, y) < 112.5) {
                x_dir = unedited_image_matrix.at<uchar>(x + 1, y );
                y_dir = unedited_image_matrix.at<uchar>(x - 1, y );
            } // Angle = 135 degrees
            else if (67.5 <= direction_matrix.at<uchar>(x, y) < 112.5) {
                x_dir = unedited_image_matrix.at<uchar>(x - 1, y - 1);
                y_dir = unedited_image_matrix.at<uchar>(x + 1, y + 1);
            }

            if ((unedited_image_matrix.at<uchar>(x, y) >= (x_dir)) && (unedited_image_matrix.at<uchar>(x, y) >= (y_dir))) {
                edited_image_matrix.at<uchar>(x, y) = unedited_image_matrix.at<uchar>(x, y);
            }
            else {
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


//
IntensityImage* applyDoubleThreshold(const IntensityImage& image, const uchar strong=60, const uchar weak=20) {

    // Make a matrix for the 'old' image
    cv::Mat unedited_image_matrix;
    
    // Convert intensity image to values for the matrix, this is done by reference.
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(image, unedited_image_matrix);

    // Create a matrix initialized to 0 of the same size of the original gradient intensity matrix
    cv::Mat edited_image_matrix;
    edited_image_matrix.create(unedited_image_matrix.rows, unedited_image_matrix.cols, CV_8UC1);

    for (int x = 1; x < unedited_image_matrix.rows - 1; x++) {
        for (int y = 1; y < unedited_image_matrix.cols - 1; y++) {
            if (unedited_image_matrix.at<uchar>(x, y) >= strong) {
                edited_image_matrix.at<uchar>(x, y) = 255;
            } 
            else if(unedited_image_matrix.at<uchar>(x, y) >= weak ){
                edited_image_matrix.at<uchar>(x, y) = 128;
            }
            else {
                edited_image_matrix.at<uchar>(x, y) = 0;
            }
        }
    }

    // Make a new intensity image for the new converted image
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();

    // Convert new matrix to new intensityimage
    HereBeDragons::NoWantOfConscienceHoldItThatICall(edited_image_matrix, *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("double_threshold.png"));
#endif // SAVE_IMAGE
 
    // return the new intensityimage
    return edited_intensity_image;
}


IntensityImage* applyHysteresisThreshold(const IntensityImage& image) {
    
    // Make a matrix for the 'old' image
    cv::Mat image_matrix;

    // Convert intensity image to values for the matrix, this is done by reference.
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(image, image_matrix);

 

    for (int x = 1; x < image_matrix.rows - 1; x++) {
        for (int y = 1; y < image_matrix.cols - 1; y++) {
            if (image_matrix.at<uchar>(x, y) == 128) {
                if (
                    image_matrix.at<uchar>(x + 1, y - 1) == 255 ||
                    image_matrix.at<uchar>(x + 1, y) == 255 ||
                    image_matrix.at<uchar>(x + 1, y + 1) == 255 ||
                    image_matrix.at<uchar>(x, y - 1) == 255 ||
                    image_matrix.at<uchar>(x, y + 1) == 255 ||
                    image_matrix.at<uchar>(x - 1, y - 1) == 255 ||
                    image_matrix.at<uchar>(x - 1, y) == 255 ||
                    image_matrix.at<uchar>(x - 1, y + 1) == 255
                    ) {
                    image_matrix.at<uchar>(x, y) = 255;
                }
                else {
                    image_matrix.at<uchar>(x, y) = 0;
                }
            }
        }
    }

    // Make a new intensity image for the new converted image
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();

    // Convert new matrix to new intensityimage
    HereBeDragons::NoWantOfConscienceHoldItThatICall(image_matrix, *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("final.png"));
#endif // SAVE_IMAGE

    // return the new intensityimage
    return edited_intensity_image;
}

IntensityImage* invertImage(const IntensityImage& original) {

    // Make a matrix for the 'old' image
    cv::Mat image_matrix;

    // Convert intensity image to values for the matrix, this is done by reference.
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(original, image_matrix);

    cv::Mat edited_image_matrix;
    edited_image_matrix.create(image_matrix.rows, image_matrix.cols, CV_8UC1);

    for (int x = 0; x < image_matrix.rows; x++) {
        for (int y = 0; y < image_matrix.cols; y++) {
            if (image_matrix.at<uchar>(x, y) == 0) {
                edited_image_matrix.at<uchar>(x, y) = 255;
            }
            else {
                edited_image_matrix.at<uchar>(x, y) = 0;
            }
        }
    }

    // Make a new intensity image for the new converted image
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();

    // Convert new matrix to new intensityimage
    HereBeDragons::NoWantOfConscienceHoldItThatICall(edited_image_matrix, *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("invert.png"));
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
//#define CANNY
 
    const double sigma = 1.4;

#ifdef CANNY
    // Apply Gaussian Blurrr
   
    auto image_after_gaussian = applyGaussianBlur(image, sigma);

    // Apply Sobell Filter
    auto image_after_sobell = applySobel(*image_after_gaussian);

    // Apply Edge Thinning
    auto image_after_edge_thinning = applyEdgeThinning(*image_after_sobell.first, image_after_sobell.second);

    // Apply Double threshold
    auto image_after_double_threshold = applyDoubleThreshold(*image_after_edge_thinning);

    // Apply Hesteresis threshols
    auto final_image = applyHysteresisThreshold(*image_after_double_threshold);

    auto invert = invertImage(*final_image);

    return invert;

#else
    cv::Mat laplacianMat;
    cv::Mat image_after_laplacian;

    auto image_after_gaussian = applyGaussianBlur(image, sigma);
    HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(*image_after_gaussian, laplacianMat);


    cv::Laplacian(laplacianMat, image_after_laplacian, CV_8U, 5, 8, 0, cv::BORDER_DEFAULT);
    
    IntensityImage* edited_intensity_image = ImageFactory::newIntensityImage();

    // Convert new matrix to new intensityimage
    HereBeDragons::NoWantOfConscienceHoldItThatICall(image_after_laplacian, *edited_intensity_image);

#ifdef SAVE_IMAGE
    ImageIO::saveIntensityImage(*edited_intensity_image, ImageIO::getDebugFileName("final.png"));
#endif // SAVE_IMAGE

    // return the new intensityimage
    return edited_intensity_image;
    
#endif // CANNY
}

IntensityImage * StudentPreProcessing::stepThresholding(const IntensityImage &image) const {
	return nullptr; 
}