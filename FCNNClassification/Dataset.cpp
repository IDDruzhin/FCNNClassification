#include "stdafx.h"
#include "Dataset.h"

Dataset::Dataset()
{
	loaded_ = false;
}

Dataset::Dataset(string train_input_path, string train_output_path, string test_input_path, string test_output_path)
{
	loaded_ = false;
}


Dataset::~Dataset()
{
}

int Dataset::GetInputSize()
{
	return input_size_;
}

int Dataset::GetOutputSize()
{
	return output_size_;
}

bool Dataset::IsLoaded()
{
	return loaded_;
}

vector<vector<ubyte>>& Dataset::GetTrainInputs()
{
	return train_inputs_;
}

vector<vector<ubyte>>& Dataset::GetTestInputs()
{
	return test_inputs_;
}

vector<unsigned char>& Dataset::GetTrainOutputs()
{
	return train_outputs_;
}

vector<unsigned char>& Dataset::GetTestOutputs()
{
	return test_outputs_;
}

vector<int>& Dataset::GetIndexes()
{
	return indexes_;
}

void Dataset::Shuffle()
{
	random_shuffle(indexes_.begin(), indexes_.end());
}

vector<ubyte> Dataset::LoadSample(string sample_path)
{
	return vector<ubyte>();
}
