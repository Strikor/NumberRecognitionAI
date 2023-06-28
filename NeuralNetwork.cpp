// All NN model functions for MNIST dataset
// Jason Graham
// 6/18/2023

#include "NeuralNetwork.h"
#include "IDXfile_Import.h"
#include <vector>
#include <random>

Network::Network(const int layers, const std::vector<int>& nodes, const float learn){
    //Establish Hyper Parameters
    this->layers = layers;
    this->nodes = nodes;
    learnRate = learn;

//Initialize weights and biases from random distribution
    //Random device for initialization of weights and biases
	std::random_device rd;
	std::mt19937 gen(rd()); //Mersenne twister algorithm
	std::uniform_real_distribution<> dist(-0.1, 0.1);

    //resize weights and biases first layer
    weights.resize(layers - 1);
    biases.resize(layers - 1);

    //resize other layers and initialize
        //Weights deals with the intermediate connective layers
        //Bias only deals with the individual activation neurons.
    for(int i = 0; i < layers - 1; ++i) { // 1 less layer because I'm dealing with the intermediate layers
        weights[i].resize(nodes[i]);
        biases[i].resize(nodes[i + 1]);
        for(int j = 0; j < nodes[i]; j++) { // the starting layer
            weights[i].resize(nodes[i + 1]);
        }

        //-----------------------------------------------------------------------------------

        // Randomly initialize weights and biases using a random normal distribution, seeded by a Mersenne twister algorithm
		for (size_t j = 0; j < weights[i].size(); ++j) { // Starting node
			for (size_t k = 0; k < weights[i][j].size(); ++k) { //Ending node
				weights[i][j][k] = dist(gen);
			}
		}
		for (size_t j = 0; j < biases[i].size(); ++j) {
			biases[i][j] = dist(gen);
		}
	}

    
    
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
