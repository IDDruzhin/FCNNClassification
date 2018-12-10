#include "stdafx.h"
#include "FCNeuralNet.h"

FCNeuralNet::FCNeuralNet()
{
}

FCNeuralNet::FCNeuralNet(vector<int> neurons_count, vector<int> activation_functions, int epochs_count, double learning_rate)
{
	srand(777);
	epochs_count_ = epochs_count;
	learning_rate_ = learning_rate;
	neurons_.resize(neurons_count.size());
	for (int i = 0; i < neurons_.size(); i++)
	{
		neurons_[i].resize(neurons_count[i]);
	}
	for (int a : activation_functions)
	{
		switch (a)
		{
		case 0:
			activation_functions_.push_back(SIGMOID);
			break;
		case 1:
			activation_functions_.push_back(HYPERBOLIC_TANGENT);
			break;
		default:
			activation_functions_.push_back(SIGMOID);
		}
	}
	weights_.resize(neurons_.size()-1);
	free_weights_.resize(weights_.size());
	deltas_.resize(weights_.size());
	for (int i = 0; i < neurons_.size()-1; i++)
	{
		weights_[i].resize(neurons_[i].size() * neurons_[i+1].size());
		free_weights_[i].resize(neurons_[i+1].size());
		deltas_[i].resize(neurons_[i+1].size());
	}
	InitWeights();
}

FCNeuralNet::~FCNeuralNet()
{
}

double FCNeuralNet::Fit(Dataset* dataset)
{
	for (int i = 0; i < epochs_count_; i++)
	{
		cout << "Epoch #" << i << endl;
		dataset->Shuffle();
		Train(dataset->GetTrainInputs(), dataset->GetTrainOutputs(), dataset->GetIndexes());
		Test(dataset->GetTestInputs(), dataset->GetTestOutputs());
		cout << "Acc = " << test_accuracy_ << endl;
	}
	return Test(dataset->GetTestInputs(), dataset->GetTestOutputs());
}

double FCNeuralNet::GetTestAccuracy()
{
	return test_accuracy_;
}

int FCNeuralNet::Predict(vector<ubyte> sample)
{
	SingleSampleCalculation(sample);
	auto output_pos = max_element(neurons_.back().begin(), neurons_.back().end());
	return distance(neurons_.back().begin(), output_pos);
	/*
	SingleSampleCalculation(sample);
	for (int i = 0; i < neurons_.back().size(); i++)
	{
		cout << i << "/" << neurons_.back()[i] << " ";
	}
	*/
}

void FCNeuralNet::InitWeights()
{
	#pragma omp parallel for
	for (int i = 0; i < weights_.size(); i++)
	{
		for (int j = 0; j < weights_[i].size(); j++)
		{
			weights_[i][j] = -0.5 + (rand() % 1000) / 1000.0;
		}
		for (int j = 0; j < free_weights_[i].size(); j++)
		{
			free_weights_[i][j] = -0.5 + (rand() % 1000) / 1000.0;
		}
	}
}

void FCNeuralNet::SoftMax()
{
	double sum = 0.0;
	for (auto &value : neurons_.back())
	{
		value = exp(value);
		sum += value;
	}
	for (auto &value : neurons_.back())
	{
		value/=sum;
	}
}

void FCNeuralNet::Sigmoid(int layer)
{
	#pragma omp parallel for
	for (int i = 0; i < neurons_[layer].size(); i++)
	{
		neurons_[layer][i]= 1.0 / (1.0 + exp(-neurons_[layer][i]));
	}
}

void FCNeuralNet::HyperbolicTangent(int layer)
{
	#pragma omp parallel for
	for (int i = 0; i < neurons_[layer].size(); i++)
	{
		neurons_[layer][i] = tanh(neurons_[layer][i]);
	}
}

void FCNeuralNet::Calculate()
{
	for (int k = 0; k < neurons_.size() - 1; k++)
	{
		#pragma omp parallel for
		for (int j = 0; j < neurons_[k+1].size(); j++)
		{
			neurons_[k+1][j] = free_weights_[k][j];
			for (int i = 0; i < neurons_[k].size(); i++)
			{
				neurons_[k + 1][j] += neurons_[k][i] * weights_[k][j*neurons_[k].size() + i];
			}
		}
		if (k + 1 != neurons_.size() - 1)
		{
			switch (activation_functions_[k])
			{
			case SIGMOID:
				Sigmoid(k + 1);
				break;
			case HYPERBOLIC_TANGENT:
				HyperbolicTangent(k + 1);
				break;
			}
		}
		else
		{
			SoftMax();
		}
	}	
}

void FCNeuralNet::SingleSampleCalculation(vector<ubyte>& input)
{
	#pragma omp parallel for
	for (int i = 0; i < neurons_[0].size(); i++)
	{
		neurons_[0][i] = input[i]/255.0;
	}
	Calculate();
}

void FCNeuralNet::CalculateDeltas(int output_class)
{
	for (int j = 0; j < deltas_.back().size(); j++)
	{
		if (j == output_class)
		{
			deltas_.back()[j] = neurons_.back()[j] - 1;
		}
		else
		{
			deltas_.back()[j] = neurons_.back()[j];
		}
	}
	for (int k = deltas_.size() - 2; k >= 0; k--)
	{
		#pragma omp parallel for
		for (int i = 0; i < deltas_[k].size(); i++)
		{
			deltas_[k][i] = 0.0f;
			for (int j = 0; j < deltas_[k + 1].size(); j++)
			{
				deltas_[k][i] += deltas_[k + 1][j] * weights_[k + 1][j * neurons_[k + 1].size() + i];
			}
			switch (activation_functions_[k])
			{
			case SIGMOID:
				deltas_[k][i] *= neurons_[k + 1][i] * (1.0 - neurons_[k + 1][i]);
				break;
			case HYPERBOLIC_TANGENT:
				deltas_[k][i] *= (1.0 + neurons_[k + 1][i]) * (1.0 - neurons_[k + 1][i]);
				break;
			}
		}
	}
}

void FCNeuralNet::BackPropogation(int output_class)
{
	CalculateDeltas(output_class);
	for (int k = 0; k < weights_.size(); k++)
	{
		#pragma omp parallel for
		for (int j = 0; j < neurons_[k + 1].size(); j++)
		{
			for (int i = 0; i < neurons_[k].size(); i++)
			{
				weights_[k][j*neurons_[k].size() + i] -= learning_rate_ * deltas_[k][j] * neurons_[k][i];
			}
			free_weights_[k][j] -= learning_rate_ * deltas_[k][j];
		}
	}
}

void FCNeuralNet::Train(vector<vector<ubyte>>& train_inputs, vector<ubyte>& train_outputs, vector<int>& indexes)
{
	for (int i = 0; i < indexes.size(); i++)
	{
		SingleSampleCalculation(train_inputs[indexes[i]]);
		BackPropogation(train_outputs[indexes[i]]);
	}
}

double FCNeuralNet::Test(vector<vector<ubyte>>& test_inputs, vector<ubyte>& test_outputs)
{
	double sum = 0.0;
	double corrects = 0.0;
	auto output_pos = max_element(neurons_.back().begin(), neurons_.back().end());
	for (int i=0; i<test_outputs.size(); i++)
	{
		SingleSampleCalculation(test_inputs[i]);
		sum -= log(neurons_.back()[test_outputs[i]]);
		output_pos = max_element(neurons_.back().begin(), neurons_.back().end());
		if (distance(neurons_.back().begin(), output_pos) == test_outputs[i])
		{
			corrects += 1.0;
		}
	}
	sum /= test_outputs.size();
	test_cross_entropy_ = sum;
	test_accuracy_ = corrects / test_outputs.size();
	return sum;
}

