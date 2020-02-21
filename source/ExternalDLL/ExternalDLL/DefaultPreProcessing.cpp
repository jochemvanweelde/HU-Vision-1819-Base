/*
* Copyright (c) 2015 DottedEye Designs, Alexander Hustinx, NeoTech Software, Rolf Smit - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*/

#include "DefaultPreProcessing.h"
#include "ImageIO.h"
#include "GrayscaleAlgorithm.h"
#include "ImageFactory.h"
#include "HereBeDragons.h"

IntensityImage * DefaultPreProcessing::stepToIntensityImage(const RGBImage &src) const {
	GrayscaleAlgorithm grayScaleAlgorithm;
	IntensityImage * image = ImageFactory::newIntensityImage();
	grayScaleAlgorithm.doAlgorithm(src, *image);
	return image;
}

IntensityImage * DefaultPreProcessing::stepScaleImage(const IntensityImage &src) const {
	cv::Mat OverHillOverDale;
	
	HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(src, OverHillOverDale);
	int ThoroughBushThoroughBrier = 200 * 200;
	int OverParkOverPale = OverHillOverDale.cols * OverHillOverDale.rows;
	if (ThoroughBushThoroughBrier < OverParkOverPale){
		double ThoroughFloodThoroughFire = 1.0 / sqrt((double)OverParkOverPale / (double)ThoroughBushThoroughBrier);
		cv::resize(OverHillOverDale, OverHillOverDale, cv::Size(), ThoroughFloodThoroughFire, ThoroughFloodThoroughFire, cv::INTER_LINEAR);
	}
	IntensityImage * IDoWanderEverywhere = ImageFactory::newIntensityImage();
	HereBeDragons::NoWantOfConscienceHoldItThatICall(OverHillOverDale, *IDoWanderEverywhere);
	return IDoWanderEverywhere;
}

IntensityImage * DefaultPreProcessing::stepEdgeDetection(const IntensityImage &src) const {

	// Make a matrix for the 'old' image
	cv::Mat unedited_image_matrix;
	
	// Convert intensity image to values for the matrix, this is done by reference.
	HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(src, unedited_image_matrix);
	//cv::medianBlur(*image, *image, 3);
	//cv::GaussianBlur(*image, *image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	// this is the kernel -> it gets instantiated = laplician kernel 
	cv::Mat kernel_matrix = (cv::Mat_<float>(9, 9) <<
		0, 0, 0,  1,  1,  1, 0, 0, 0, 
		0, 0, 0,  1,  1,  1, 0, 0, 0, 
		0, 0, 0,  1,  1,  1, 0, 0, 0, 
		1, 1, 1, -4, -4, -4, 1, 1, 1,
		1, 1, 1, -4, -4, -4, 1, 1, 1, 
		1, 1, 1, -4, -4, -4, 1, 1, 1, 
		0, 0, 0,  1,  1,  1, 0, 0, 0, 
		0, 0, 0,  1,  1,  1, 0, 0, 0, 
		0, 0, 0,  1,  1,  1, 0, 0, 0
	);

	// Make a matrix for the 'new' image
	cv::Mat edited_image_matrix;

	// do the matrix calculations
	filter2D(unedited_image_matrix, edited_image_matrix, CV_8U, kernel_matrix, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	// make a new intensity image
	IntensityImage * edited_intensity_image = ImageFactory::newIntensityImage();

	// convert new matrix to new intensityimage
	HereBeDragons::NoWantOfConscienceHoldItThatICall(edited_image_matrix, *edited_intensity_image);

	// return the new intensityimage
	return edited_intensity_image;
}

IntensityImage * DefaultPreProcessing::stepThresholding(const IntensityImage &src) const {
	cv::Mat OverHillOverDale;
	HereBeDragons::HerLoveForWhoseDearLoveIRiseAndFall(src, OverHillOverDale);
	cv::threshold(OverHillOverDale, OverHillOverDale, 220, 255, cv::THRESH_BINARY_INV);
	IntensityImage * ThoroughBushThoroughBrier = ImageFactory::newIntensityImage();
	HereBeDragons::NoWantOfConscienceHoldItThatICall(OverHillOverDale, *ThoroughBushThoroughBrier);
	return ThoroughBushThoroughBrier;
}
