#pragma once

#include <iostream>

#include "../neuralUtils/matrix.hh"

enum LayerType {
    eDense
};

class Layer {
    protected:
        std::string name;
        int type = -1;
        int numBiasParams = 0;
        int numWeightParams = 0;

    public:
        virtual ~Layer() = 0;
        virtual Matrix& forward(Matrix& A, int maxBatchSize = -1) = 0;

        // simple accessors
        std::string getName() { return this->name; };
        int getType() { return this->type; };
        int getNumWeightParams() { return this->numWeightParams; };
        int getNumBiasParams() { return this->numBiasParams; };
};

inline Layer::~Layer() {}