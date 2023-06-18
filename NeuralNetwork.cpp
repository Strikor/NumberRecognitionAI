// All NN model functions for MNIST dataset
// Jason Graham
// 6/18/2023

#include "NeuralNetwork.h"
#include "IDXfile_Import.h"
#include <vector>

Network::Network(const int layers, const std::vector<int>& nodes, const float learn){
    //Establish Hyper Parameters
    numLayers = layers;
    numNodes = nodes;
    learnRate = learn;

    //Initialize weights and biases
}

int Network::learn() {
    //loop
        //Create a batch (epoch)
        //Pass each batch to feedforward()
        //Take cost of previous batch (how far off of expected)
        //Use cost for backpropagate()
}

std::vector<float> Network::feedforward(std::vector<float> image){
    //Run through the network and retrieve an output
    //calculate and return final layer activation
}

int Network::backpropagate(const std::vector<float> activations, std::vector<float> expected){ //Output activations & one-hot encoded vector
    //Take the activations from the feedforward and compare to one-hots
    //Calculate cost for each output node
    //Work backwards
}
