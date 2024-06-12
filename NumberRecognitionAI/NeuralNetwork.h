#pragma once
// All NN model functions for MNIST dataset
// Jason Graham
// 6/7/2023

#include <vector>
#include "IDXfile_Import.h"

class Network {
public:
    Network(const int layers, const std::vector<int>& nodes, const float learn);
    void learn(int batchSize, int epochs); //The actual learning cycle
    std::vector<std::vector<std::vector<float>>> feedforward(const std::vector<std::vector<float>>& batch);
    std::vector<float> calculateCost(const std::vector<std::vector<std::vector<float>>> activations, const std::vector<float> expected); //Cost function
    int backpropagate(const std::vector<std::vector<std::vector<float>>>& allActivations, const std::vector<std::vector<float>>& expectedOutput,
                        const std::vector<std::vector<std::vector<float>>>& weights, const std::vector<std::vector<float>>& biases);
    void updateWeights(const std::vector<std::vector<std::vector<float>>>& weightUpdates, const std::vector<std::vector<float>>& biasUpdates);

private:
    int                 layers; //Total Layers
    std::vector<int>    nodes;  //Per Layer
    float               learnRate; 
    int                 epoch;  //Iteration through total dataset
    int                 batchSize;
    IDXfile             data;   //Data to be used for training and testing
    IDXfile             lable;  //lables for data

//Node Specific Variables
    //[connectionLayer][startNode][endNode]weight
        //weights are dependant on the intermediate layer instead of the nodes
    std::vector<std::vector<std::vector<float>>> weights;

    //[layer - 1][node]bias
        //biases are tied to a specific neuron instead of the intermediate layer
	std::vector<std::vector<float>> biases;

    std::vector<int> createRandomIndexes(int numIndexes, int maxIndex);




};