#pragma once
#include "Dataset.h"
#include <omp.h>
#include <iostream>

class FCNeuralNet
{

typedef
enum ACTIVATION_FUNCTION_TYPE
{
	SIGMOID = 0,
	HYPERBOLIC_TANGENT = SIGMOID + 1
} 	ACTIVATION_FUNCTION_TYPE;

public:
	FCNeuralNet();
	FCNeuralNet(vector<int> neurons_count, vector<int> activation_functions, int epochs_count, double learning_rate);
	~FCNeuralNet();
	double Fit(Dataset* dataset);
	double GetTestAccuracy();
	int Predict(vector<ubyte> sample);
private:
	int epochs_count_;
	double learning_rate_;
	vector<vector<double>> neurons_;
	vector<vector<double>> weights_;
	vector<vector<double>> free_weights_;
	vector<vector<double>> deltas_;
	vector<ACTIVATION_FUNCTION_TYPE> activation_functions_;

	double test_cross_entropy_;
	double test_accuracy_;

	void InitWeights();
	void SoftMax();
	void Sigmoid(int layer);
	void HyperbolicTangent(int layer);
	void Calculate();
	void SingleSampleCalculation(vector<ubyte>& input);
	void CalculateDeltas(int output_class);
	void BackPropogation(int output_class);
	void Train(vector<vector<ubyte>>& train_inputs, vector<ubyte>& train_outputs, vector<int>& indexes);
	double Test(vector<vector<ubyte>>& test_inputs, vector<ubyte>& test_outputs);
};

