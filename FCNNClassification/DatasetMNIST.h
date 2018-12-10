#pragma once
#include "Dataset.h"
class DatasetMNIST :
	public Dataset
{
public:
	DatasetMNIST(string train_input_path, string train_output_path, string test_input_path, string test_output_path, int output_size);
	~DatasetMNIST();
};

