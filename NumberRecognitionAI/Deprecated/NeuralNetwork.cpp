// Class containing the primary functions for running the training for NN
// Jason Graham
// 4/5/2023

#include "NeuralNetwork.h"
#include <algorithm>

//First vector controls the number of layers as well as the deapth of those layers
//Second is epochs, how many times the network trains through the entire dataset
//Third is learning rate, controls how aggressively the loss function adjusts the weights and biases
//Forth is batch sizes, controls how many samples are used before the equations are adjusted
//	The weight and biases adjustment is what is computationally expensive
NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, const int epoch, const float learnRate, const int batch)
	: layer_sizes(layers), numEpoch(epoch), learningRate(learnRate), batchSize(batch) {

	//Adjust the vectors to account for the amount of layers
	weights.resize(layer_sizes.size() - 1);
	biases.resize(layer_sizes.size() - 1);

	//Random device for initialization of weights and biases
	std::random_device rd;
	std::mt19937 gen(rd()); //Mersenne twister algorithm
	std::uniform_real_distribution<> dist(-0.1, 0.1);

	for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
		weights[i].resize(layer_sizes[i]); //Second dimension represents starting node
		for (size_t j = 0; j < layer_sizes[i]; ++j) {
			weights[i][j].resize(layer_sizes[i + 1]); // Third dimension represents ending node
		}

		biases[i].resize(layer_sizes[i + 1]); // Only the neurons actually have activation biases as input has none

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

	// Size activations and weighted sums correctly
	activations.resize(layer_sizes.size());
	weighted_sums.resize(layer_sizes.size() - 1);

	for (size_t i = 0; i < layer_sizes.size(); ++i) {
		activations[i].resize(layer_sizes[i]); // Contains the value of a node
		if (i < layer_sizes.size() - 1) {
			weighted_sums[i].resize(layer_sizes[i + 1]); // Contains the pre-activated/normalized value
		}
	}

};

std::vector<float> NeuralNetwork::forward_propagation(const std::vector<float>& input) {
	//Set the first layers node values to the input
	activations[0] = input;

	//Calculate weighted sums for the program
	for (size_t i = 1; i < layer_sizes.size(); ++i) {//Which Layer's nodes we're working on
		//Iterate through the layer's nodes
		for (size_t j = 0; j < layer_sizes[i]; ++j) {
			weighted_sums[i - 1][j] = biases[i - 1][j]; //At this point (i, j) represents a node in the current unknown layer adding that nodes specific

			for (size_t k = 0; k < layer_sizes[i - 1]; ++k) { //Looks into previous layer and begins looking at all connections between the previous and current layers.
				weighted_sums[i - 1][j] += activations[i - 1][k] * weights[i - 1][k][j];

			}

			activations[i][j] = std::max(0.0f, weighted_sums[i - 1][j]); //Applies ReLU function, IMPORTANT: while this function only creates positive slopes the next weight can make it negative

		}
	}

	return activations.back(); //returns the last layers activations
};

float compute_cost(const std::vector<float>& predictions, const std::vector<float>& labels) {
	float cost = 0.0;
	size_t n = predictions.size();
	
	for (size_t i = 0; i < n; ++i) {
		float diff = predictions[i] - labels[i];
		cost += diff * diff;
	}

	cost /= n;
	return cost;
}
