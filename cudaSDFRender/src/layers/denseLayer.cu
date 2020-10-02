#include "denseLayer.hh"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>


// Aliases
using ColumnMajor = cutlass::layout::ColumnMajor;
using ColumnMajorTransposed = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm60;
using OpClass = cutlass::arch::OpClassSimt;

using relu_op = cutlass::epilogue::thread::LinearCombinationRelu<float,1>;
using linear_op = cutlass::epilogue::thread::LinearCombination<float,1>;

// Use Gemm defaults (for now...)
using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<OpClass, //op class
                                                                ArchTag, // arch tag
                                                                float,  // element a
                                                                float,  // element b
                                                                float,  // element c
                                                                float>; // element accum

using ThreadblockShape = cutlass::gemm::GemmShape<8,32,8>;
using WarpShape = cutlass::gemm::GemmShape<8,16,8>;


using GemmRelu = cutlass::gemm::device::Gemm<
                                                float,                      // Data-type of A matrix
                                                ColumnMajorTransposed,      // Layout of A matrix
                                                float,                      // Data-type of B matrix
                                                ColumnMajor,                // Layout of B matrix
                                                float,                      // Data-type of C matrix
                                                ColumnMajor,                // Layout of C matrix                                                             
                                                float,                      // Element Accumulator Type
                                                OpClass,
                                                ArchTag,
                                                ThreadblockShape, 
                                                WarpShape, 
                                                DefaultConfig::InstructionShape,
                                                relu_op>;

using GemmLinear = cutlass::gemm::device::Gemm<
                                                float, ColumnMajorTransposed,  
                                                float, ColumnMajor,     
                                                float, ColumnMajor,                                                                 
                                                float,        
                                                OpClass,
                                                ArchTag,
                                                ThreadblockShape, 
                                                WarpShape, 
                                                DefaultConfig::InstructionShape,
                                                linear_op>;

using BatchedGemmRelu = cutlass::gemm::device::GemmBatched<
                                                float, ColumnMajorTransposed,
                                                float, ColumnMajor,
                                                float, ColumnMajor,
                                                float, 
                                                OpClass,
                                                ArchTag,
                                                ThreadblockShape, 
                                                WarpShape, 
                                                DefaultConfig::InstructionShape,
                                                relu_op>;


using BatchedGemmLinear = cutlass::gemm::device::GemmBatched<
                                                float, ColumnMajorTransposed,
                                                float, ColumnMajor,
                                                float, ColumnMajor,
                                                float,
                                                OpClass,
                                                ArchTag,
                                                ThreadblockShape, 
                                                WarpShape, 
                                                DefaultConfig::InstructionShape,
                                                linear_op>;


cudaError_t denseLayerForward(
    float* W, float* A, float* Z, float* b, 
    int M, int N, int K,
    int activation) {

    cutlass::cutStatus status;

    if (activation == ReLU) {
        GemmRelu gemm;
        // Construct the CUTLASS GEMM arguments object.
        GemmRelu::Arguments args({M , N, K},        // Gemm Problem dimensions
            {W, K},                                 // ref for source matrix A
            {A, K},                                 // ref for source matrix B
            {b, M},                                 // ref for source matrix C
            {Z, M},                                 // ref for destination matrix D
            {1.0f, 1.0f});                          // Scalars used in the Epilogue
         status = gemm(args);
    } else {
        GemmLinear gemm;
        // Construct the CUTLASS GEMM arguments object.
        GemmLinear::Arguments args({M , N, K},
            {W, K},    
            {A, K},           
            {b, M},            
            {Z, M},             
            {1.0f, 1.0f});     
        status = gemm(args);
    }
          
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    if (status != cutlass::cutStatus::kSuccess) {
        printf("ERROR: %s\n", cutlassGetStatusString(status));
        printf("Single - M: %d N: %d K: %d\n", M,N,K);
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}

cudaError_t batchedDenseLayerForward(
    const float* W, const float* A, float* Z, const float* b, 
    int M, int N, int K, int numBatches,
    int activation)
 {

    cutlass::cutStatus status;

    if (activation == ReLU) {
        BatchedGemmRelu batchedGemm;

        status = batchedGemm({
            {M, N, K},
            {W, K}, 
            0,
            {A, K}, 
            K,        // batch in N
            {b, M}, 
            0,
            {Z, M}, 
            M,        // stride
            {1.0f, 1.0f},
            numBatches
        });
    } else {
        BatchedGemmLinear batchedGemm;

        status = batchedGemm({
            {M, N, K},
            {W, K}, 
            0,
            {A, K}, 
            K,        // stride
            {b, M}, 
            0,
            {Z, M}, 
            M,        // stride
            {1.0f, 1.0f},
            numBatches
        });
    }
    
    if (status != cutlass::cutStatus::kSuccess) {
        printf("ERROR: %s\n", cutlassGetStatusString(status));
        printf("Batched - M: %d N: %d K: %d numBatches: %d\n", M,N,K, numBatches);
        return cudaErrorUnknown;
    }

    return cudaSuccess;

 }


// Dense Layer Class imp
DenseLayer::DenseLayer(
    std::string name, 
    std::vector<std::vector<float>> weights, 
    std::vector<float> biases,
    int activation, bool hostOnly
) 
{
    this->W = Matrix(Shape(weights.size(), weights[0].size()), hostOnly);
    this->numWeightParams = weights.size() * weights[0].size();
    this->b = Matrix(Shape(biases.size(), 1), hostOnly); // (1,B)
    this->numBiasParams = biases.size();
    this->name = name;
    this->type = eDense;
    this->hostOnly = hostOnly;

    b.allocateMemory();
    W.allocateMemory();
    
    initializeBias(biases);
    initializeWeights(weights);

    this->activation = activation;
}

DenseLayer::~DenseLayer()
{ }

void DenseLayer::initializeBias(std::vector<float> biases) {
    for (int x = 0; x < biases.size(); x ++) {
        b[x] = biases[x];
    }

    if (!hostOnly) {
        b.copyHostToDevice();
    }
}

void DenseLayer::initializeWeights(std::vector<std::vector<float>> weights) {
    for (int x = 0; x < weights.size(); x++) {
        for (int y = 0; y < weights[0].size(); y ++) {
            W[y*W.shape.x + x] = weights[x][y]; //ROW MAJOR!
        }
    }

    if (!hostOnly) {
        W.copyHostToDevice();
    }
}

Matrix& DenseLayer::forward(Matrix& A, int maxBatchSize) {
    assert(W.shape.x == A.shape.x);

    this->A = A;

    Shape Z_shape;
    if (maxBatchSize == -1) {
        Z_shape = Shape(W.shape.y, A.shape.y);
    } else {
        Z_shape = Shape(W.shape.y, maxBatchSize);
    }
    
    Z.maybeAllocateMemory(Z_shape);

    cudaError_t ok = computeAndStoreLayerOutput(A);
    checkCudaErrors(ok);

    Z.shape.y = A.shape.y;

    return Z;   
}

cudaError_t DenseLayer::computeAndStoreLayerOutput(Matrix& A) {
    cudaError_t ok;

    if (A.shape.y == 1) {
        // single item
        ok = denseLayerForward(
            W.deviceData.get(), A.deviceData.get(), Z.deviceData.get(), b.deviceData.get(), 
            W.shape.y, // transpose!
            b.shape.y, 
            W.shape.x, 
            this->activation
        );
    } else {
        // batched !
        ok = batchedDenseLayerForward(
            (const float*)W.deviceData.get(), (const float*)A.deviceData.get(), Z.deviceData.get(), (const float*)b.deviceData.get(), 
            W.shape.y,  //M
            b.shape.y,  //N
            W.shape.x,  //K
            A.shape.y,  //Num Batches
            this->activation
        );
    }

    checkCudaErrors(ok);

    return cudaSuccess;
}

int DenseLayer::getXDim() const {
	return W.shape.x;
}

int DenseLayer::getYDim() const {
	return W.shape.y;
}

Matrix DenseLayer::getWeightsMatrix() const {
	return W;
}

Matrix DenseLayer::getBiasVector() const {
	return b;
}

