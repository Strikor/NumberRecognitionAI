#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <atomic>



class Display {
public:
	static void visualizeActivations(const std::vector<std::vector<float>>& activations);

	static std::atomic<bool> shouldEnd;
};