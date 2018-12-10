#include "stdafx.h"
#include "FCNeuralNet.h"

FCNeuralNet::FCNeuralNet()
{
}

FCNeuralNet::FCNeuralNet(vector<int> neurons_count, vector<int> activation_functions, int epochs_count, double learning_rate)
{
	srand(777);
	//srand(123);
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
	/*
	input_count_ = input_count;
	hidden_count_ = hidden_count+1;
	output_count_ = output_count;
	input_neurons_.resize(input_count_);
	input_neurons_[0] = 1.0;
	hidden_neurons_.resize(hidden_count_);
	hidden_neurons_[0] = 1.0;
	output_neurons_.resize(output_count_);
	first_weights_.resize(input_count_*(hidden_count_-1));
	second_weights_.resize(hidden_count_*output_count_);
	*/
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
	/*
	for (auto &weight : weights_)
	{
		//weight = -0.1 + 0.2*((double)rand() / RAND_MAX);
		//weight = -0.25 + (rand() % 50) / 100.0;
		//weight = -0.1 + (rand() % 10) / 100.0;
		//weight = 0.0;
		weight = -0.5 + (rand() % 1000) / 1000.0;
		//weight = (rand() % 100) / 1000.0;
	}
	for (auto &weight : second_weights_)
	{
		//weight = -0.1 + 0.2*((double)rand() / RAND_MAX);
		//weight = -0.25 + (rand() % 50) / 100.0;
		//weight = -0.5 + (rand() % 100) / 100.0;
		//weight = -0.1 + (rand() % 10) / 100.0;
		//weight = 0.0;
		//weight = (rand() % 100) / 1000.0;
		weight = -0.5 + (rand() % 1000) / 1000.0;
	}
	*/
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
	/*
	#pragma omp parallel for
	for (int i = 1; i < hidden_count_; i++)
	{
		hidden_neurons_[i] = 0.0;
		for (int j = 0; j < input_count_; j++)
		{
			hidden_neurons_[i] += input_neurons_[j] * first_weights_[(i-1)*input_count_ + j];
		}
	}
	switch (activation_function_)
	{
	case SIGMOID:
		Sigmoid();
		break;
	case HYPERBOLIC_TANGENT:
		HyperbolicTangent();
		break;
	}
	#pragma omp parallel for
	for (int i = 0; i < output_count_; i++)
	{
		output_neurons_[i] = 0.0;
		for (int j = 0; j < hidden_count_; j++)
		{
			output_neurons_[i] += hidden_neurons_[j] * second_weights_[i*hidden_count_ + j];
		}
	}
	SoftMax();
	*/
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
	/*
	double tmp;
	double tmp1;
	#pragma omp parallel for private(tmp,tmp1)
	for (int i = 1; i < hidden_count_; i++)
	{
		switch (activation_function_)
		{
		case SIGMOID:
			tmp = hidden_neurons_[i] * (1.0 - hidden_neurons_[i]) * learning_rate_;
			break;
		case HYPERBOLIC_TANGENT:
			tmp = (1.0 + hidden_neurons_[i]) * (1.0 - hidden_neurons_[i]) * learning_rate_;
			break;
		}
		for (int j = 0; j < input_count_; j++)
		{
			//tmp = tmp*input_neurons_[j]*learning_rate_;
			tmp1 = tmp * input_neurons_[j];
			for (int l = 0; l < output_count_; l++)
			{
				if (l == output_class)
				{
					//first_weights_[(i - 1)*input_count_ + j] += (1.0 - output_neurons_[l])*second_weights_[l*hidden_count_ + i]*tmp;
					first_weights_[(i - 1)*input_count_ + j] += (1.0 - output_neurons_[l])*second_weights_[l*hidden_count_ + i] * tmp1;
				}
				else
				{
					//first_weights_[(i - 1)*input_count_ + j] += (-output_neurons_[l])*second_weights_[l*hidden_count_ + i]*tmp;
					first_weights_[(i - 1)*input_count_ + j] += (-output_neurons_[l])*second_weights_[l*hidden_count_ + i] * tmp1;
				}
				
			}
		}
	}
	#pragma omp parallel for private(tmp)
	for (int i = 0; i < output_count_; i++)
	{
		if (output_class == i)
		{
			tmp = (1.0 - output_neurons_[i])*learning_rate_;
		}
		else
		{
			tmp = (-output_neurons_[i])*learning_rate_;
		}
		for (int j = 0; j < hidden_count_; j++)
		{
			second_weights_[i*hidden_count_ + j] += tmp*hidden_neurons_[j];		
		}
	}
	*/
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

