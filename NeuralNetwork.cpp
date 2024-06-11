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
    lable = IDXfile("train-labels.idx1-ubyte");
}

void Network::learn(int batchSize) {
    //loop
    int correctCount = 0;
    for (int i = 0; i < data.getNumItems(); ++i) {
        //Figure out what the ideal batch size is and what I should use for it
        std::vector<std::vector<float>> images;
        std::vector<std::vector<float>> labels = lable.getLabels();

        for (int j = 0; j < data.getNumItems(); ++j) {
            images.push_back(data.getImage());
        }

        //Randomly shuffle the images vector
        std::vector<int> batches = createRandomIndexes(images.size(), images.size() - 1);

        for (int k = 0; k < batches.size(); k += batchSize) {
            std::vector<std::vector<float>> batch;
            for (int l = 0; l < batchSize; ++l) {
            batch.push_back(images[batches[k + l]]);
            }

            //Pass each batch to feedforward()
            std::vector<std::vector<std::vector<float>>> activations = feedforward(batch);

            //Take cost of previous batch (how far off of expected)
            std::vector<std::vector<float>> expected;
            for (int m = 0; m < batchSize; ++m) {
                expected.push_back(labels[batches[k + m]]);
            }

            backpropagate(activations, expected);
            /*int predictedIndex = 
       
            // Check if the prediction is correct
            if (predictedIndex == expected[0]) {
            correctCount++;
            std::cout << "Test " << i << " - Correct prediction! Expected: " << expected[0] << ", Predicted: " << predictedIndex << std::endl;
            } else {
            std::cout << "Test " << i << " - Incorrect prediction! Expected: " << expected[0] << ", Predicted: " << predictedIndex << std::endl;
            }*/
        }
    }

    // Calculate success rate
    float successRate = static_cast<float>(correctCount) / data.getNumItems();
    std::cout << "Success rate of the current epoch: " << successRate * 100 << "%" << std::endl;
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

int Network::backpropagate(const std::vector<std::vector<std::vector<float>>> &allActivations, const std::vector<std::vector<float>> &expectedOutput) {
    // Initialize weight and bias updates
    std::vector<std::vector<std::vector<float>>> weightUpdates(weights.size());
    std::vector<std::vector<float>> biasUpdates(biases.size());

    // Calculate updates for each image in the batch
    for (int n = 0; n < allActivations.size(); ++n) {
        const std::vector<std::vector<float>> &activations = allActivations[n];

        // Calculate output errors
        std::vector<float> errors(activations.back().size());
        for (int i = 0; i < activations.back().size(); ++i) {
            errors[i] = activations.back()[i] - expectedOutput[n][i];
        }

        // Backpropagate errors and calculate updates
        for (int i = layers - 2; i >= 0; --i) {
            std::vector<float> newErrors(nodes[i]);
            weightUpdates[i].resize(nodes[i]);
            for (int j = 0; j < nodes[i]; ++j) {
                weightUpdates[i][j].resize(nodes[i + 1]);
                for (int k = 0; k < nodes[i + 1]; ++k) {
                    float update = learnRate * errors[k] * activations[i][j];
                    weightUpdates[i][j][k] += update;
                    newErrors[j] += errors[k] * weights[i][j][k];
                }
            }
            errors = newErrors;
            biasUpdates[i].resize(nodes[i + 1]);
            for (int j = 0; j < nodes[i + 1]; ++j) {
                biasUpdates[i][j] += learnRate * errors[j];
            }
        }

        // Check if the prediction is correct
        int predictedIndex = std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
        int expectedIndex = std::distance(expectedOutput[n].begin(), std::max_element(expectedOutput[n].begin(), expectedOutput[n].end()));
        if (predictedIndex == expectedIndex) {
            std::cout << "Prediction: " << predictedIndex << ", Expected: " << expectedIndex << " - Correct" << std::endl;
        } else {
            std::cout << "Prediction: " << predictedIndex << ", Expected: " << expectedIndex << " - Incorrect" << std::endl;
        }
    }

    // Average updates and apply them to weights and biases
    for (int i = 0; i < layers - 1; ++i) {
        for (int j = 0; j < nodes[i]; ++j) {
            for (int k = 0; k < nodes[i + 1]; ++k) {
                weights[i][j][k] -= weightUpdates[i][j][k] / allActivations.size();
            }
        }
        for (int j = 0; j < nodes[i + 1]; ++j) {
            biases[i][j] -= biasUpdates[i][j] / allActivations.size();
        }
    }

    // Calculate percent correct for the batch
    int correctCount = 0;
    for (int n = 0; n < allActivations.size(); ++n) {
        const std::vector<std::vector<float>> &activations = allActivations[n];
        int predictedIndex = std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
        int expectedIndex = std::distance(expectedOutput[n].begin(), std::max_element(expectedOutput[n].begin(), expectedOutput[n].end()));
        if (predictedIndex == expectedIndex) {
            correctCount++;
        }
    }
    float percentCorrect = static_cast<float>(correctCount) / allActivations.size() * 100;
    std::cout << "Percent correct for the batch: " << percentCorrect << "%" << std::endl;

    return 0;
}

std::vector<std::vector<std::vector<float>>> Network::feedforward(std::vector<std::vector<float>> batch){
    // Run through the network and retrieve an output
    // Calculate and return all layer activations for each image

    std::vector<std::vector<std::vector<float>>> allActivations;

    for(int i = 0; i < batch.size(); ++i) {
        std::vector<std::vector<float>> activations;
        std::vector<float> activation = batch[i];
        activations.push_back(activation); // Store input layer activations
        for(int j = 0; j < layers - 1; ++j) {
            std::vector<float> newActivations;
            for(int k = 0; k < nodes[j + 1]; ++k) {
                float sum = 0;
                for(int l = 0; l < nodes[j]; ++l) {
                    sum += activation[l] * weights[j][l][k];
                }
                sum += biases[j][k];
                // Apply activation function
                float activatedSum = (sum > 0) ? sum : 0;
                newActivations.push_back(activatedSum);
            }
            activation = newActivations;
            activations.push_back(activation); // Store this layer's activations
        }
        allActivations.push_back(activations);
    }
    
    return allActivations;
}
std::vector<float> Network::calculateCost(std::vector<std::vector<std::vector<float>>> activations, std::vector<float> expected){
    //Calculate the cost of the activations
    //Return the cost
    std::vector<float> costs;
    for(int i = 0; i < activations.size(); ++i) {
        float cost = 0;
        for(int j = 0; j < activations[i][activations[i].size() - 1].size(); ++j) {
            cost += (activations[i][activations[i].size() - 1][j] - expected[j]) * (activations[i][activations[i].size() - 1][j] - expected[j]);
        }
        costs.push_back(cost);
    }
    return costs;
}
