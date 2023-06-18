#pragma once
// All NN model functions for MNIST dataset
// Jason Graham
// 6/7/2023

#include <vector>

class Network {
public:
    Network(const int layers, const std::vector<int>& nodes, const float learn);
    int learn(); //The actual learning cycle
    std::vector<float> feedforward(const std::vector<float> image); //Output consists of one-hot encoded vector //Output is cost
    int backpropagate(const std::vector<float> activations, std::vector<float> expected);

private:
    int                 numLayers; //Total Layers
    std::vector<int>    numNodes;  //Per Layer
    float               learnRate; 
    int                 numEpoch;  //Iteration through total dataset

    //Node Specific Variables
    //[connectionLayer][startNode][endNode]
    std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<std::vector<float>>> biases;




};
