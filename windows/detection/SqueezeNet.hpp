#pragma once
#include "detect.hpp"
class SqueezeNet :
	public Detector
{
public:
	std::vector<vector<float> > Detect(const cv::Mat& img);
};

