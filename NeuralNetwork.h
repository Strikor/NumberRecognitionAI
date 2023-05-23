#pragma once
// Header file for NN class
// Jason Graham
// 4/5/2023

#include <vector>
#include <random>


class NeuralNetwork {
public:
	NeuralNetwork(const std::vector<int>& layer_sizes, int epoch, float learnRate, int batch);
	std::vector<float> forward_propagation(const std::vector<float>& input);
	float compute_cost(const std::vector<float>& predictions, const std::vector<float>& labels);
	void backpropagation(const std::vector<float>& input, const std::vector<float>& labels);
	void update_weights_biases(float learning_rate);
	void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& labels, int epochs, int batch_size, float learning_rate);
	float evaluate(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& labels);


	//(const std::vector<int>& layer_sizes = { 784, 16, 16, 10 }, int epoch = 20, float learnRate = 0.1f, int batch = 32)

private:
	int numEpoch;
	float learningRate;
	int batchSize;


	//Rafactor code to have weights be a 3d vector to be able to contail [Weight Layer][Starting Node][Ending Node]
	std::vector<int> layer_sizes;
	std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<float>> biases;
	std::vector<std::vector<float>> weight_gradients;
	std::vector<std::vector<float>> bias_gradients;
	std::vector<std::vector<float>> weighted_sums;
	std::vector<std::vector<float>> activations;

};
