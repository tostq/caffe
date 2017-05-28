#pragma once
#include "detect.hpp"
class SSD :
	public Detector
{
public:
	std::vector<vector<float> > Detect(const cv::Mat& img);
};

