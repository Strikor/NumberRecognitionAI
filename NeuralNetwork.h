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
    int                 layers; //Total Layers
    std::vector<int>    nodes;  //Per Layer
    float               learnRate; 
    int                 epoch;  //Iteration through total dataset

//Node Specific Variables
    //[connectionLayer][startNode][endNode]weight
        //weights are dependant on the intermediate layer instead of the nodes
    std::vector<std::vector<std::vector<float>>> weights;

    //[layer - 1][node]bias
        //biases are tied to a specific neuron instead of the intermediate layer
	std::vector<std::vector<float>> biases;




};