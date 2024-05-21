// All NN model functions for MNIST dataset
// Jason Graham
// 6/18/2023

#include "NeuralNetwork.h"
#include "IDXfile_Import.h"
#include <vector>
#include <set>
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

//Grab data from file
    data = IDXfile("train-images.idx3-ubyte");
    lable = IDXfile("train-labels.idx1-ubyte");
    
}

void Network::learn(int batchSize) {
    //loop
            //Figure out what the ideal batch size is and what I should use for it

        //Pass each batch to feedforward()
            //How should I do this?
            
        //Take cost of previous batch (how far off of expected)
        //Use cost for backpropagate()

    //[Image][pixel values] value
    std::vector<std::vector<float>> images;
    std::vector<std::vector<float>> labels = lable.getLabels();

    for(int i = 0; i < data.getNumItems(); ++i) {
        images.push_back(data.getImage());
    }

    //Randomly shuffle the images vector
    std::vector<int> batches = createRandomIndexes(images.size(), images.size() - 1);

    for(int i = 0; i < batches.size(); i += batchSize) {
        std::vector<std::vector<float>, std::vector<int>> batch;
        for(int j = 0; j < batchSize; ++j) {
            batch.push_back(images[batches[i + j]], labels[batches[i + j]]);
        }

        feedforward(batch);
    }
    
    

    
}

std::vector<int> createRandomIndexes(int numIndexes, int maxIndex) {
    std::vector<int> result;
    std::set<int> seen;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, maxIndex);

    while (result.size() < numIndexes) {
        int index = dis(gen);
        if (seen.count(index) == 0) {
            seen.insert(index);
            result.push_back(index);
        }
    }

    return result;
}

int Network::backpropagate(const std::vector<float> activations, std::vector<float> expected){ //Output activations & one-hot encoded vector
    //Take the activations from the feedforward and compare to one-hots
    //Calculate cost for each output node
    //Work backwards
}

std::vector<float> Network::feedforward(std::vector<std::vector<float>> batch, std::vector<std::vector<float>> labels){
    //Run through the network and retrieve an output
    //calculate and return final layer activation

    std::vector<float> costs(10);

    for(int i = 0; i < batch.size(); ++i) {
        std::vector<float> activations = batch[i];
        for(int j = 0; j < layers - 1; ++j) {
            std::vector<float> newActivations;
            for(int k = 0; k < nodes[j + 1]; ++k) {
                float activation = 0;
                for(int l = 0; l < nodes[j]; ++l) {
                    activation += activations[l] * weights[j][l][k];
                }
                activation += biases[j][k];
                newActivations.push_back(activation);
            }
            activations = newActivations;
        }


        std::vector<float> costs = calculateCost(activations, labels[i]); // Calculate cost for each batch
        // Do something with the cost
    }
    
}

std::vector<float> Network::calculateCost(std::vector<float> activations, std::vector<float> expected){
    //Calculate the cost of the activations
    //Return the cost
    std::vector<float> costs;
    for(int i = 0; i < activations.size(); ++i) {
        costs.push_back((activations[i] - expected[i]) * (activations[i] - expected[i]));
    }

    
}

int Network::backpropagate(const std::vector<float> activations, std::vector<float> expected){ //Output activations & one-hot encoded vector
    //Take the activations from the feedforward and compare to one-hots
    //Calculate cost for each output node
    //Work backwards
}
