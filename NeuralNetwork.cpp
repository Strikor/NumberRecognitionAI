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

    //resize weights and biases for each layer
    weights.resize(layers - 1);
    biases.resize(layers - 1);

    //resize other layers and initialize
    for(int i = 0; i < layers - 1; ++i) {
        weights[i].resize(nodes[i]);
        biases[i].resize(nodes[i + 1]);
        for(int j = 0; j < nodes[i]; j++) {
            weights[i][j].resize(nodes[i + 1]);
        }

        // Randomly initialize weights and biases using a random normal distribution, seeded by a Mersenne twister algorithm
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] = dist(gen);
            }
        }
        for (size_t j = 0; j < biases[i].size(); ++j) {
            biases[i][j] = dist(gen);
        }
    }

    //Grab data from file
    data = IDXfile("train-images.idx3-ubyte");
    label = IDXfile("train-labels.idx1-ubyte");
}

void Network::learn(int batchSize) {
    //loop
    for (int i = 0; i < data.getNumItems(); ++i) {
        //Figure out what the ideal batch size is and what I should use for it
        std::vector<std::vector<float>> images;
        std::vector<std::vector<float>> labels = label.getLabels();

        for (int i = 0; i < data.getNumItems(); ++i) {
            images.push_back(data.getImage());
        }

        //Randomly shuffle the images vector
        std::vector<int> batches = createRandomIndexes(images.size(), images.size() - 1);

        for (int i = 0; i < batches.size(); i += batchSize) {
            std::vector<std::vector<float>> batch;
            for (int j = 0; j < batchSize; ++j) {
                batch.push_back(images[batches[i + j]]);
            }

            //Pass each batch to feedforward()
            std::vector<std::vector<float>> activations = feedforward(batch, labels);

            //Take cost of previous batch (how far off of expected)
            std::vector<float> expected;
            for (int j = 0; j < batchSize; ++j) {
                expected.push_back(labels[batches[i + j]][0]);
            }
            std::vector<float> costs = calculateCost(activations, expected);

            //Use cost for backpropagate()
            backpropagate(activations, expected);
        }
    }
}

std::vector<int> Network::createRandomIndexes(int numIndexes, int maxIndex) {
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

int Network::backpropagate(const std::vector<float>& activations, const std::vector<float>& expected) {
    // Take the activations from the feedforward and compare to one-hots
    // Calculate cost for each output node
    std::vector<float> errors(activations.size());
    for (int i = 0; i < activations.size(); ++i) {
        errors[i] = activations[i] - expected[i];
    }

    // Update weights and biases
    for (int i = layers - 2; i >= 0; --i) {
        for (int j = 0; j < nodes[i]; ++j) {
            for (int k = 0; k < nodes[i + 1]; ++k) {
                weights[i][j][k] -= learnRate * errors[k] * activations[j];
            }
            biases[i][j] -= learnRate * errors[j];
        }
    }

    // Work backwards
    std::vector<float> newErrors(nodes[layers - 2]);
    for (int j = 0; j < nodes[layers - 2]; ++j) {
        float error = 0;
        for (int k = 0; k < nodes[layers - 1]; ++k) {
            error += errors[k] * weights[layers - 2][j][k];
        }
        newErrors[j] = error;
    }
    errors = newErrors;

    // Return the index of the output node with the highest activation
    int maxIndex = 0;
    float maxActivation = activations[0];
    for (int i = 1; i < activations.size(); ++i) {
        if (activations[i] > maxActivation) {
            maxActivation = activations[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

std::vector<std::vector<float>> Network::feedforward(std::vector<std::vector<float>> batch, std::vector<std::vector<float>> labels){
    //Run through the network and retrieve an output
    //calculate and return final layer activation

    std::vector<std::vector<float>> activations;

    for(int i = 0; i < batch.size(); ++i) {
        std::vector<float> activation = batch[i];
        for(int j = 0; j < layers - 1; ++j) {
            std::vector<float> newActivations;
            for(int k = 0; k < nodes[j + 1]; ++k) {
                float sum = 0;
                for(int l = 0; l < nodes[j]; ++l) {
                    sum += activation[l] * weights[j][l][k];
                }
                sum += biases[j][k];
                // Apply activation function (e.g., sigmoid)
                float activatedSum = sigmoid(sum);
                newActivations.push_back(activatedSum);
            }
            activation = newActivations;
        }
        activations.push_back(activation);
    }
    
    return activations;

}
std::vector<float> Network::calculateCost(std::vector<std::vector<float>> activations, std::vector<float> expected){
    //Calculate the cost of the activations
    //Return the cost
    std::vector<float> costs;
    for(int i = 0; i < activations.size(); ++i) {
        float cost = 0;
        for(int j = 0; j < activations[i].size(); ++j) {
            cost += (activations[i][j] - expected[j]) * (activations[i][j] - expected[j]);
        }
        costs.push_back(cost);
    }
    return costs;
}

float Network::calculateAverageCost(const std::vector<float>& costs) {
    float sum = 0;
    for (float cost : costs) {
        sum += cost;
    }
    return sum / costs.size();
}       for(int j = 0; j < activations[i].size(); ++j) {
            cost += (activations[i][j] - expected[j]) * (activations[i][j] - expected[j]);
        }
        costs.push_back(cost);
    }
    return costs;
}
