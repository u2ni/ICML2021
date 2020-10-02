#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"

#include <iostream>
#include <time.h>
#include <math.h>


bool loadModelFromH5 (std::string fp, NeuralNetwork& nn, bool hostOnly) {
    HighFive::File file(fp, HighFive::File::ReadOnly);

    std::vector<std::string> kerasLayers = file.listObjectNames();
    int layerCount = 0;

    for (std::vector<std::string>::iterator it = kerasLayers.begin() ; it != kerasLayers.end(); ++it) {
        // for each layer, copy weights to eigen
        HighFive::ObjectType objType = file.getObjectType(*it);

        if (objType != HighFive::ObjectType::Group) {
            std::cout << "Unsupported Layer\n";
            return false;
        }

        HighFive::Group group = file.getGroup(*it);
        int n = group.getNumberObjects();
        
        if (n != 1) {
            std::cout << "Unsupported Layer\n";
            return false;
        }

        group = group.getGroup(*it);
        std::vector<std::string> matNames = group.listObjectNames();

        std::vector<std::vector<float>> weights;
        std::vector<float> biases;

        for (std::vector<std::string>::iterator matIt = matNames.begin(); matIt != matNames.end(); ++matIt) {
            objType = group.getObjectType(*matIt);
            if (objType != HighFive::ObjectType::Dataset) {
                std::cout << "Unsupported Layer\n";
                return false;
            }

            // parse the weights and biases
            HighFive::DataSet dataset = group.getDataSet(*matIt);
            std::vector<size_t> dim = dataset.getDimensions();

            if (dim.size() == 1) {
                dataset.read(biases);
            } else if (dim.size() == 2) {
                dataset.read(weights);
            }
            else {
                std::cout << "Unsupported layer, to many dims!\n";
                return false;
            }
        }

        int activation = ReLU; //RELU
        if  ((it != kerasLayers.end()) && (next(it) == kerasLayers.end())) {
            activation = Tanh; 
        }

        nn.addLayer(new DenseLayer(
            std::string("Dense_") + std::to_string(layerCount), 
            weights, 
            biases, 
            activation,     
            hostOnly            // only allocate on host!
        ));
        layerCount ++;
    }
    return true;
}

bool singleTest(int batchSize) {
    NeuralNetwork nn;
    bool ok = loadModelFromH5("model.h5", nn, false);

    printf("Testing single inference\n");

    Matrix Y;
    Matrix X = Matrix(Shape(3,1));

    X.allocateMemory();

    X[0] = static_cast<float>(0.1);
    X[1] = static_cast<float>(0.2);
    X[2] = static_cast<float>(0.3);

    X.copyHostToDevice();

    clock_t start = std::clock();
    for (int i = 0; i < batchSize; i ++) {
        
        Y = nn.forward(X);
    }
    cudaDeviceSynchronize();
    std::cout << "Took: " << (std::clock() - start)/(double)(CLOCKS_PER_SEC / 1000) <<" ms for "<< batchSize << " inferences\n";
    

    Y.copyDeviceToHost();

    printf("(%f %f %f): %f \n",X[0],X[1], X[2], tanh(Y[0]));
}

void batchTest(int batchSize, bool doVerify) {
    NeuralNetwork nn;
    bool ok = loadModelFromH5("model.h5", nn, false);

    printf("\n\nTesting Batched inference (Batchsize: %d)\n\n", batchSize);
    Matrix Y;
    Matrix X = Matrix(Shape(3,batchSize));

    X.allocateMemory();
    
    for (int i = 0; i < batchSize*3; i ++) {
        X[i] = 0.0;
    }

    X.copyHostToDevice();

    clock_t start = std::clock();
    Y = nn.forward(X);
    cudaDeviceSynchronize();

    std::cout << "Took: " << (std::clock() - start)/(double)(CLOCKS_PER_SEC / 1000) <<" ms for "<< batchSize << " inferences\n";
    
    // assert that they are all the same!
    if (doVerify) {
        printf("Checking for errors...\n");
        Y.copyDeviceToHost();
        float first = Y[0];
        for (int i = 1; i < Y.shape.y; i ++){
            if (Y[i] != first) {
                printf("ERROR: %f\n", Y[i]);
                return;
            }
        }
        printf("Woah there aren't any!! All evaluated (%f,%f,%f):%f\n", X[0], X[1],X[2], Y[0]);
    }
}

void streamedBatchedTest(int batchSize, bool doVerify) {
    NeuralNetwork nn;
    bool ok = loadModelFromH5("model.h5", nn, false);

    printf("\n\nTesting Batched inference (Batchsize: %d)\n\n", batchSize);
    Matrix Y;
    Matrix X = Matrix(Shape(3,batchSize));

    X.allocateMemory();
    
    for (int i = 0; i < batchSize*3; i ++) {
        X[i] = 0.0;
    }

    X.copyHostToDevice();

    clock_t start = std::clock();
    Y = nn.forward(X);
    cudaDeviceSynchronize();

    std::cout << "Took: " << (std::clock() - start)/(double)(CLOCKS_PER_SEC / 1000) <<" ms for "<< batchSize << " inferences\n";
    
    // assert that they are all the same!
    if (doVerify) {
        printf("Checking for errors...\n");
        Y.copyDeviceToHost();
        float first = Y[0];
        for (int i = 1; i < Y.shape.y; i ++){
            if (Y[i] != first) {
                printf("ERROR: %f\n", Y[i]);
                return;
            }
        }
        printf("Woah there aren't any!! All evaluated (%f,%f,%f):%f\n", X[0], X[1],X[2], Y[0]);
    }
}

int main () {
    //

    int batchSize = 1000000;
    bool doVerify = true;
    //singleTest(batchSize);
    batchTest(batchSize, doVerify);
   
}

