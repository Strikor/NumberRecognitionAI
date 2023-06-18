#pragma once
// Simple Class for reading in IDX type files
// Jason Graham
// 1/24/2023

#include <iostream>
#include <fstream>
#include <vector>
//#include <filesystem>

class IDXfile {
public:
	IDXfile() {};
	IDXfile(std::string);

	std::vector<float> getImage() {
		std::vector<float> ret;
		for (int i = 0; i < rows * cols; ++i) {
			ret.push_back(data[curByte]);
			++curByte;
		}
		++curImage;

		return ret;
	}

	const std::vector<float> getData() { return data; }
	const std::vector<std::vector<float>> getLabels() { return one_hot_encoded_labels; }
	const std::string getPath() { return path; }
	const int getMagic() { return magicNum; }
	const int getNumItems() { return numItems; }
	const int getRows() { return rows; }
	const int getCols() { return cols; }

private:
	std::string	path = "";
	std::vector<float> data; //Used only for files containing image data or some other data
	std::vector<std::vector<float>> one_hot_encoded_labels; //Used only for the labels
	int curByte    = 0;
	int curImage   = 0;
	int magicNum   = 0;
	int numItems   = 0;
	int rows	   = 0;
	int cols	   = 0;

	
};
