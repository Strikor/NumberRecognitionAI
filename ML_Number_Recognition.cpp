// The main driver file for the MNIST Neural Network
// Jason Graham
// 1/24/2023

#include <iostream>
#include "IDXfile_import.h"

int main()
{
    std::cout << "Hello World!\n";

    IDXfile testing = IDXfile("../TrainingData/train-images.idx3-ubyte");

    std::vector<float> image = testing.getImage();
    int rows = testing.getRows();
    int cols = testing.getCols();

    //Print's letter
    for (int i = 0; i < rows * cols; ++i) {
        if (image[i] != 0) std::cout << "1 ";
        else std::cout << "0 ";
        if ((i + 1) % rows == 0) std::cout << std::endl;
    }

    std::cout << std::endl;

    image = testing.getImage();
    //prints letter
    for (int i = 0; i < rows * cols; ++i) {
        if (image[i] != 0) std::cout << "1 ";
        else std::cout << "0 ";
        if ((i + 1) % rows == 0) std::cout << std::endl;
    }

    std::cout << testing.getMagic() << std::endl;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
