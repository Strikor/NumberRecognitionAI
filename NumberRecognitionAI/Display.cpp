#include "Display.h"
#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <atomic>


std::atomic<bool> Display::shouldEnd = false;

// Function to visualize the neural network activations
void Display::visualizeActivations(const std::vector<std::vector<float>>& activations) {
    const int windowWidth = 800;
    const int windowHeight = 600;
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Neural Network Activations");
    window.setFramerateLimit(30);

    while (window.isOpen()) {
        sf::Event event;
        bool polled = window.pollEvent(event);
        if (polled) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        if (!polled) { // No event was polled, so let's sleep for a brief time to prevent high CPU usage
            sf::sleep(sf::milliseconds(100));
        }

        if (shouldEnd) { // Check if shouldEnd was triggered during the sleep
            window.close();
        }

        window.clear(sf::Color::Blue); // Clear the screen with black

        int xOffset = 10; // Start 10 pixels from the left
        for (size_t layerIndex = 0; layerIndex < activations.size(); ++layerIndex) {
            const auto& layer = activations[layerIndex];
            int nodesInLayer = layer.size();

            if (layerIndex == 0) { // First layer as 28x28 grid
                int size = 28;
                int gap = 2; // Gap between squares
                int squareSize = (windowHeight - (size + 1) * gap) / size; // Calculate the square size

                for (int i = 0; i < nodesInLayer; ++i) {
                    int x = i % size;
                    int y = i / size;
                    sf::RectangleShape square(sf::Vector2f(squareSize, squareSize));
                    square.setPosition(xOffset + x * (squareSize + gap), y * (squareSize + gap));
                    float intensity = layer[i];
                    square.setFillColor(sf::Color(255 * intensity, 255 * intensity, 255 * intensity)); // Set fill color based on activation
                    window.draw(square);
                }
                xOffset += size * (squareSize + gap) + 10; // Move to the right for the next layer
            }
            else { // Other layers as single columns
                int squareSize = 20; // Fixed size for nodes in column
                int gap = 5; // Vertical gap between nodes
                for (int i = 0; i < nodesInLayer; ++i) {
                    sf::RectangleShape square(sf::Vector2f(squareSize, squareSize));
                    square.setPosition(xOffset, i * (squareSize + gap));
                    float intensity = layer[i];
                    square.setFillColor(sf::Color(255 * intensity, 255 * intensity, 255 * intensity)); // Set fill color
                    window.draw(square);
                }
                xOffset += squareSize + 10; // Move to the right for the next layer
            }
        }

        window.display();
    }
}