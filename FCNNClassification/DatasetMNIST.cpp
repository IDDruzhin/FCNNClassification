#include "stdafx.h"
#include "DatasetMNIST.h"


DatasetMNIST::DatasetMNIST(string train_input_path, string train_output_path, string test_input_path, string test_output_path, int output_size)
{
	output_size_ = output_size;
	ifstream f;
	int tmp;
	int size;
	f.open(train_input_path, ios::binary);
	if (!f.is_open())
	{
		return;
	}
	f.read((char*)(&tmp), sizeof(int));
	f.read((char*)(&size), sizeof(int));
	size = _byteswap_ulong(size);
	train_inputs_.resize(size);
	train_outputs_.resize(size);
	indexes_.resize(size);
	iota(indexes_.begin(), indexes_.end(), 0);
	f.read((char*)(&tmp), sizeof(int));
	input_size_ = _byteswap_ulong(tmp);
	f.read((char*)(&tmp), sizeof(int));
	input_size_ *= _byteswap_ulong(tmp);
	for (int i = 0; i < size; i++)
	{
		train_inputs_[i].resize(input_size_);
		f.read((char*)(&train_inputs_[i][0]), sizeof(ubyte)*input_size_);
	}
	f.close();

	f.open(train_output_path, ios::binary);
	if (!f.is_open())
	{
		return;
	}
	f.seekg(8);
	f.read((char*)(&train_outputs_[0]), sizeof(ubyte)*size);
	f.close();

	f.open(test_input_path, ios::binary);
	if (!f.is_open())
	{
		return;
	}
	f.read((char*)(&tmp), sizeof(int));
	f.read((char*)(&size), sizeof(int));
	size = _byteswap_ulong(size);
	test_inputs_.resize(size);
	test_outputs_.resize(size);
	f.seekg(16);
	for (int i = 0; i < size; i++)
	{
		test_inputs_[i].resize(input_size_);
		f.read((char*)(&test_inputs_[i][0]), sizeof(ubyte)*input_size_);
	}
	f.close();

	f.open(test_output_path, ios::binary);
	if (!f.is_open())
	{
		return;
	}
	f.seekg(8);
	f.read((char*)(&test_outputs_[0]), sizeof(ubyte)*size);
	f.close();
	loaded_ = true;
}


DatasetMNIST::~DatasetMNIST()
{
}
