#pragma once
#include "Dataset.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class DatasetMNIST :
	public Dataset
{
public:
	DatasetMNIST(string train_input_path, string train_output_path, string test_input_path, string test_output_path, int output_size);
	~DatasetMNIST();
	vector<ubyte> LoadSample(string sample_path);
};

