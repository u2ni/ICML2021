#include "matrix.hh"

Matrix::Matrix(size_t x_dim, size_t y_dim, bool hostOnly) :
    shape(x_dim, y_dim), deviceData(nullptr), hostData(nullptr),
    deviceAllocated(false), hostAllocated(false), hostOnly(hostOnly)
{ }

Matrix::Matrix(Shape shape, bool hostOnly) :
    Matrix(shape.x, shape.y, hostOnly)
{ }

void Matrix::allocateDeviceMemory() {
    if (!deviceAllocated) {
        cudaError_t ok;
        float * deviceMemory = nullptr;

        ok = cudaMalloc(&deviceMemory, shape.x * shape.y * sizeof(float));
        checkCudaErrors(ok);
        deviceData = std::shared_ptr<float> (deviceMemory, [&](float* ptr){ cudaFree(ptr); });
        deviceAllocated = true;
    }
}

void Matrix::allocateHostMemory() {
    if (!hostAllocated) {
        hostData = std::shared_ptr<float> (new float[shape.x*shape.y], [&](float* ptr){ delete[] ptr; });
        hostAllocated = true;
    }
}

void Matrix::allocateMemory() {

    allocateHostMemory();
    
    if (!hostOnly) {
        allocateDeviceMemory();
    }
}

void Matrix::maybeAllocateMemory(Shape shape) {
    if (!deviceAllocated && !hostAllocated) {
        this->shape = shape;
        allocateMemory();
    } 
}

void Matrix::copyHostToDevice() {
    if (deviceAllocated && hostAllocated) {
        cudaError_t ok;
        ok = cudaMemcpy(deviceData.get(), hostData.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaErrors(ok);
    } else {
        printf("Failed to copy from host to device... nothing initialized\n");
    }
}

void Matrix::copyDeviceToHost() {
    if (deviceAllocated && hostAllocated) {
        cudaError_t ok;
        ok = cudaMemcpy(
            hostData.get(), 
            deviceData.get(), 
            shape.x * shape.y * sizeof(float), 
            cudaMemcpyDeviceToHost
        );

        checkCudaErrors(ok);

    } else {
        printf("Failed to copy from device to host... nothing initialized\n");
    }
}

float& Matrix::operator[](const int index) {
	return hostData.get()[index];
}

const float& Matrix::operator[](const int index) const {
	return hostData.get()[index];
}

