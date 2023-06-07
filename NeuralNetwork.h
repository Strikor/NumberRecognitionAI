#pragma once
// All NN model functions for MNIST dataset
// Jason Graham
// 6/7/2023

#include <vector>

class Network {
public:
    Network(const std::vector<int>& layer_sizes, int epoch, float learnRate, int batch);
    int feedforward(std::vector<float> image, std::vector<float> output); //Output consists of one-hot encoded vector
    int backpropagate();

private:
    int                 numLayers;
    std::vector<int>    numNodes;
    float               learnRate;
    int                 numEpoch;

    //Node Specific Variables
    std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;




};
