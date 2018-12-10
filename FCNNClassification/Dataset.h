#pragma once
typedef unsigned char ubyte;

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>

using namespace std;

class Dataset
{
public:
	Dataset();
	Dataset(string train_input_path, string train_output_path, string test_input_path, string test_output_path, int output_size);
	virtual ~Dataset() = 0;
	int GetInputSize();
	int GetOutputSize();
	bool IsLoaded();
	vector<vector<ubyte>>& GetTrainInputs();
	vector<vector<ubyte>>& GetTestInputs();
	vector<ubyte>& GetTrainOutputs();
	vector<ubyte>& GetTestOutputs();
	vector<int>& GetIndexes();
	void Shuffle();
protected:
	bool loaded_;
	int input_size_;
	int output_size_;
	vector<vector<ubyte>> train_inputs_;
	vector<ubyte> train_outputs_;
	vector<vector<ubyte>> test_inputs_;
	vector<ubyte> test_outputs_;
	vector<int> indexes_;
};

