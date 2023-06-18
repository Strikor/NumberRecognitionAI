// ML_Number_Recognition.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "IDXfile_Import.h"

IDXfile::IDXfile(std::string path) : path(path) {
	std::ifstream input(path, std::ios::binary); //create the input system, std::ios::binary removes "formatting" from file making it just take however many bits are requested.

	if (!input) {
		std::cerr << "Error: Can not open " << path << std::endl;
		exit(2);
	}

	std::vector<unsigned char> bytes(
		(std::istreambuf_iterator<char>(input)), //Iterates through the entire file and fill vector with the values
		(std::istreambuf_iterator<char>())); //After the first iterator runs out it fills the rest with a blank iterator

	input.close();

	//Bit-shifting into a 32-bit integer magicNum
	magicNum = int((bytes[0]) << 24 |
		(bytes[1]) << 16 |
		(bytes[2]) << 8 |
		(bytes[3]));
	numItems = int((bytes[4]) << 24 | (bytes[5]) << 16 | (bytes[6]) << 8 | (bytes[7]));

	//Only shows up in image sets
	if (magicNum == 2051) {//Image Set
		rows = int((bytes[8]) << 24 | (bytes[9]) << 16 | (bytes[10]) << 8 | (bytes[11]));
		cols = int((bytes[12]) << 24 | (bytes[13]) << 16 | (bytes[14]) << 8 | (bytes[15]));
		curByte = 16;

		// Resize the normalized data vector
		data.resize(bytes.size() - curByte);

		// Normalize pixel values to the range [0, 1]
		for (size_t i = 0; i < bytes.size() - curByte; ++i) {
			data[i] = static_cast<float>(bytes[i + curByte]) / 255.0f;
		}
	}
	else {//Labels & other
		curByte = 8;

		//Creates a 2d vector that will contain the one-hot encoded data. Label of 4 = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}
		std::vector<std::vector<float>> one_hot_encoded_labels(bytes.size() - curByte, std::vector<float>(10, 0.0f));

		//Generates the encoded data
		for (size_t i = 0; i < bytes.size() - curByte; ++i) {
			one_hot_encoded_labels[i][bytes[i + curByte]] = 1.0f;
		}
	}

	//With header data out of the way we can reset curByte to 0 to keep track of the data vector.
	curByte = 0;

};
