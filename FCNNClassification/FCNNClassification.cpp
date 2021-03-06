// FCNNClassification.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <iostream>
#include <sstream>
#include <memory>
#include "Dataset.h"
#include "DatasetMNIST.h"
#include "FCNeuralNet.h"

using namespace std;

int main(int argc, char* argv[])
{
	vector<string> arg_names;
	arg_names.push_back("Train images path");
	arg_names.push_back("Train labels path");
	arg_names.push_back("Test images path");
	arg_names.push_back("Test labels path");
	arg_names.push_back("Number of neurons in the hidden layers. Use / as separator");
	arg_names.push_back("Activation function in the hidden layers (0 - sigmoid, 1 - tanh). Use / as separator");
	arg_names.push_back("Epochs count");
	arg_names.push_back("Learning rate");

	if (argc < 5)
	{
		cout << "Data paths must be specified" << endl;
		cout << "Command line arguments:" << endl;
		for (int i = 1; i<arg_names.size() + 1; i++)
		{
			cout << i << ". " << arg_names[i - 1] << endl;
		}
		cout << "Press Enter to exit" << endl;
		getchar();
		return 0;
	}
	string train_input_path = argv[1];
	string train_output_path = argv[2];
	string test_input_path = argv[3];
	string test_output_path = argv[4];
	vector<int> layers_sizes;
	vector<int> activation_functions;
	int epochs_count = 10;
	double learning_rate = 0.01;
	stringstream ss;
	string tmp;
	if (argc > 5)
	{
		ss << argv[5];
		while (getline(ss, tmp, '/'))
		{
			layers_sizes.push_back(atoi(tmp.c_str()));
		}
		if (argc > 6)
		{
			ss.clear();
			ss << argv[6];
			while (getline(ss, tmp, '/'))
			{
				activation_functions.push_back(atoi(tmp.c_str()));
			}
		}
		if (argc > 6)
		{
			epochs_count = atoi(argv[7]);
		}
		if (argc > 6)
		{
			learning_rate = atof(argv[8]);
		}
	}
	if (layers_sizes.size() == 0)
	{
		layers_sizes.push_back(300);
		layers_sizes.push_back(500);
		layers_sizes.push_back(100);
	}
	while (activation_functions.size() != layers_sizes.size())
	{
		activation_functions.push_back(0);
	}
	for (int i = 1; i<5; i++)
	{
		cout << i << ". " << arg_names[i - 1] << " = " << argv[i] << endl;
	}
	cout << "5. " << arg_names[4] << ": " << layers_sizes[0];
	for (int i = 1; i < layers_sizes.size(); i++)
	{
		cout << "/" << layers_sizes[i];
	}
	cout << endl;
	cout << "6. " << arg_names[5] << ": " << activation_functions[0];
	for (int i = 1; i < activation_functions.size(); i++)
	{
		cout << "/" << activation_functions[i];
	}
	cout << endl;
	cout << "7. " << arg_names[6] << " = " << epochs_count << endl;
	cout << "8. " << arg_names[7] << " = " << learning_rate << endl;
	cout << "Press Enter to continue" << endl;
	getchar();

	unique_ptr<Dataset> data = make_unique<DatasetMNIST>(train_input_path, train_output_path, test_input_path, test_output_path);
	if (!data->IsLoaded())
	{
		cout << "Unable to load data" << endl;
		cout << "Press Enter to exit" << endl;
		getchar();
		return 0;
	}
	layers_sizes.insert(layers_sizes.begin(), data->GetInputSize());
	layers_sizes.push_back(data->GetOutputSize());

	FCNeuralNet net(layers_sizes, activation_functions, epochs_count, learning_rate);
	cout << "Cross entropy=" << net.Fit(data.get()) << endl;
	cout << "Accuracy=" << net.GetTestAccuracy() << endl;
	cout << "Recognize image (Y/N)?" << endl;
	char answer;
	cin >> answer;
	while (answer != 'n' && answer != 'N')
	{
		cout << "Input image path" << endl;
		string path;
		cin >> path;
		int prediction = net.Predict(data->LoadSample(path));
		if (prediction >= 0)
		{
			cout << endl << "Image class is: " << prediction << endl;
		}		
		cout << "Again (Y/N)?" << endl;
		cin >> answer;
	}
	return 0;
}

