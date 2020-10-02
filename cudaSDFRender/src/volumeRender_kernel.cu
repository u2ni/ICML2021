#ifndef _NEURALRENDER_KERNEL_CU_
#define _NEURALRENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"
#include "neuralUtils/image.hh"

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

typedef unsigned int  uint;
typedef unsigned char uchar;


typedef struct
{
    float4 m[3];
} float3x4;

typedef struct
{
    float4 m[4];
} float4x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
__constant__ float4x4 c_normalMatrix;
__constant__ int c_coloringType;    // 0- normal ratio, 1- matcap
__constant__ int c_numInputs;       // 3- standard, 4- frame
__constant__ int c_frameNumber;     // used only in animation mode

// tetrahedron method of estimating normals (http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm)
__constant__ float tetrahedronVerts[] = {  
    1, -1, -1,
    -1, -1,  1,
    -1,  1, -1,
    1,  1,  1,
};

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

struct Sphere
{
    float3 c;   //center
    float r;    //radius
};

const uint BACKGROUND_COLOR = 0;
const int COLOR_MASK_VAL = 4;
const float NORMAL_EPSILON = 0.0001;
const float MARCHING_EPSILON = 0.000001;
const int MAX_STEPS = 6000;

// SDF KERNELS AND HELPERS! 
// Collection of signed distance operations and primitives.
// modified from Iquilez' awesome site :) https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

__device__
float sdfSphere( float3 p, float s )
{
  return length(p)-s;
}

__device__
float3 max(float3 a, float3 b)
{
  return make_float3(a.x > b.x ? a.x : b.x,
                a.y > b.y ? a.y : b.y,
                a.z > b.z ? a.z : b.z);
}

__device__
float sdfBox(float3 p, float3 b, float r = 0.0) {
    float3 q = fabs(p) - b;
    float3 zero = make_float3(0.0);
    float d = length(max(q, zero)) + min(max(q.x, max(q.y,q.z)), 0.0) - r;
    
    //return min(max(q.x,max(q.y,q.z)),0.0) + length()
    return d;
}

__device__
float sdfPlane( float3 p) {
    return p.y - 0.5;  // we force onto bottom of bounds
}

__device__
float sdfCylinder( float3 p, float3 c) {
    float l = length(make_float2(p.x, p.y) - make_float2(c.x, c.z));
  
    return l-c.y;
}

__device__ 
float sdfOpDisplace(float3 p, float s) {
    float d = s;

    d += sin(18*p.x)*sin(18*p.y)*sin(18*p.z)*0.06;

    return d;
}

__device__
float sdfOpRound(float s, float rad) {
    return s - rad;
}

__device__
float sdfOpOnion(float s, float thickness) {
    return abs(s) - thickness;
}

// some useful SDF helpers.
__device__
float sdfOpIntersect(float distA, float distB) {
    return max(distA, distB);
}

__device__
float sdfOpUnion(float distA, float distB) {
    return min(distA, distB);
}

__device__
float sdfOpSubtraction(float distA, float distB) {
    return max(distA, -distB);
}

__device__
float sdfOpSmoothSubtraction( float d1, float d2, float k ) {
    float h = __saturatef( 0.5 - 0.5*(d1+d2)/k);
    float mix = d1*(1.0-h) - d2*h;
    return mix + k*h*(1.0-h); }

__device__
float sdfOpSmoothUnion(float d1, float d2, float k) {
    float h = __saturatef(0.5 + 0.5*(d2-d1)/k);
    float mix = d2*(1.0-h) + d1*h;
    return mix - k*h*(1.0-h);
}

__device__
float displacementPattern(float3 p, float nSDF) {
    return sdfOpDisplace(p, tanh(nSDF));
}

__device__
float sdfOpBlend(float d1, float d2, float k) {
    return k*d1 + (1-k)*d2;
}

__device__
float manyCylinderCut(float3 p, float nSDF) {
    float s = nSDF;

    float3 c = make_float3(0.02);
    float3 cP = p;
    cP.y -= 0.5;
    for (int i = 0; i < 300; i ++) {
        if (i%20 == 0 ) {
            cP.y += 0.1;
            cP.x = p.x + 0.9;
        }
        
        s = sdfOpSmoothSubtraction(s, sdfCylinder(cP,c), 0.01);
        cP.x -= 0.1;
    }

    return s;
}

__device__
float manySphere(float3 p, float nSDF, bool doUnion) {
    float s = nSDF;

    float3 cP = p;
    cP.y -= 0.6;
    cP.z += -0.7 + (c_frameNumber* 2*0.7/360);
    for (int i = 0; i < 9; i ++) {
        if (i%3 == 0 ) {
            cP.y += 0.4;
            cP.x = p.x + 0.5;
        }
        if (doUnion){
            s = sdfOpSmoothUnion(s, sdfSphere(cP,0.1), 0.01);
        } else {
            s = sdfOpSmoothSubtraction(s, sdfSphere(cP,0.1), 0.01);
        }
        cP.x -= 0.4;
    }
    return s;
}


// intersect ray with a sphere
__device__
bool intersectSphere(Ray ray, Sphere sphere, float *tnear, float *tfar)
{
    float3 Q = ray.o - sphere.c;
    float a = dot(ray.d, ray.d);
    float b = 2.0 * dot(Q, ray.d);
    float c = dot(Q,Q) - sphere.r*sphere.r;
    float discrim = b*b - 4*a*c;

    if (discrim > 0) {
        *tnear = (-b - sqrt(discrim)) / (2.0 * a); 
        *tfar =  (-b + sqrt(discrim)) / (2.0 *a);
        return true;
    }
    return false;
}

__device__ 
float sceneSDF(float3 p, float nSDF) {

    //return displacementPattern(p, tanh(nSDF));
    //return sdfSphere(p, 0.9);
    //return sdfOpRound(tanh(nSDF),0.04);
    //return manySphere(p, tanh(nSDF), false);
    //return manyCylinderCut(p, nSDF);
    //float3 boxp = make_float3(p.x+0.5, p.y+0.3, p.z-0.4);
    //float3 boxb = make_float3(0.1,0.1,0.1);

    //return sdfOpSmoothUnion(sdfBox(boxp,boxb,0.01), tanh(nSDF), 0.02);

    //return sdfOpSmoothSubtraction(sdfCylinder(boxp,boxb), tanh(nSDF), 0.05);//, nSDF);

    //return sdfOpBlend(sdfBox(p,boxb), tanh(nSDF), abs(sin(c_frameNumber*M_PI/360)));

    return tanh(nSDF);
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__
float4 mul(const float4x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = dot(v, M.m[3]);
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ 
float3 getFloat3(const float* D, int id) {
    return make_float3(
        D[id],
        D[id + 1],
        D[id + 2]
    );
}

__device__ 
void setFloat3(float* D, int id, float3 f) {
    D[id] = f.x;
    D[id + 1] = f.y;
    D[id + 2] = f.z;
}


__global__ void
initMarcher(
    uint *d_output, 
    uint *d_mask,
    float *d_points,
    float *d_ray,
    float *d_tfar,
    uint imageW, 
    uint imageH, 
    int maxSteps
)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    const float3 boxMin = make_float3(-0.5f, -0.5f, -0.5f);
    const float3 boxMax = make_float3(0.5f, 0.5f, 0.5f);

    int id = y*imageW + x;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // With more shapes, this would be created elsewhere and passed in
    Sphere boundingSphere;
    boundingSphere.c = make_float3(0.0f, 0.0f, 0.0f);
    boundingSphere.r = 0.9;

    // find intersection with box
    float tnear, tfar;
    //int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    bool hit = intersectSphere(eyeRay, boundingSphere, &tnear, &tfar);

    // no need to march if ray never hits bounding primitive
    if ( !hit ) {
        d_mask[id] = 0;
        // TODO: we should run a mini sdf loop here to get all non-neural geom things showing...
        d_output[id] = BACKGROUND_COLOR;
        return;
    }

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // start ray at edge of bounds
    float3 point = eyeRay.o + eyeRay.d*tnear;
 
    //store starting position
    setFloat3(d_points, 3*id, point);
    
    //store ray (to skip comps going forward)
    setFloat3(d_ray, 3*id, eyeRay.d);
    
    //store tfar. (decremented as we march the ray), exit condition
    d_tfar[id] = tfar;

    // Stencil update
    d_mask[id] = 1;
}


__device__ 
float3 surfaceNormal(int idx, const float* d_sdf, const float3 p) {
    // Calculate surface normal using tetrahedron technique (4 pts instead of 6)
    float3 tetP = make_float3(tetrahedronVerts[0], tetrahedronVerts[1], tetrahedronVerts[2]);
    float3 p0 = tetP*sceneSDF(p + tetP*NORMAL_EPSILON, d_sdf[idx]);

    tetP = make_float3(tetrahedronVerts[3], tetrahedronVerts[4], tetrahedronVerts[5]);
    float3 p1 = tetP*sceneSDF(p + tetP*NORMAL_EPSILON, d_sdf[idx+1]);

    tetP = make_float3(tetrahedronVerts[6], tetrahedronVerts[7], tetrahedronVerts[8]);
    float3 p2 = tetP*sceneSDF(p + tetP*NORMAL_EPSILON, d_sdf[idx+2]);;

    tetP = make_float3(tetrahedronVerts[9], tetrahedronVerts[10], tetrahedronVerts[11]);
    float3 p3 = tetP*sceneSDF(p + tetP*NORMAL_EPSILON, d_sdf[idx+3]);

    return normalize(p0 + p1 + p2 + p3);
}


__device__
uint facingColor(float3 n, float3 rayDir) {
    float ratio = max(0.0, dot(n,-rayDir) );
    return rgbaFloatToInt(make_float4(ratio,ratio,ratio, 1.0));
}


__device__ 
uint matCapColor(float3 normal, const uint* d_matcap, int matW, int matH) {
    float4 normal_eye4 = mul(c_normalMatrix, make_float4(normal.x, normal.y,normal.z, 0.0));
    float3 normal_eye = normalize(make_float3(normal_eye4.x, normal_eye4.y, normal_eye4.z));
    
    // TODO: matcap should be in texture memory... 
    float fuvx = (normal_eye.x * 0.5 + 0.5) ;
    float fuvy = (normal_eye.y * 0.5 + 0.5) ;

    int uvx = int(fuvx*(matW-1));
    int uvy = int(fuvy*(matH-1));
    
    int index = uvy * matW + uvx;

    if (index < 0) {
        //fall back that should not occur.
        printf("\n\n!!!!CRAP!!!!\n\n\n");
        return rgbaFloatToInt(make_float4(0.0f));
    }

    if (d_matcap[index] == 0) {
        printf("LIKELY A BAD MATCAP! uv: (%d, %d):(%f) n: (%f,%f,%f)  ne: (%f,%f,%f)  len_n: %f\n",uvx, uvy, sqrt(pow(fuvx,2)+ pow(fuvy,2)), normal.x, normal.y, normal.z, normal_eye.x, normal_eye.y, normal_eye.z, length(normal));
    }
    return d_matcap[index];
}


__global__ void
singleMarch(
    uint* d_output,
    const uint* d_idSDFMap,
    uint* d_mask,
    const float* d_sdf,
    float* d_points,
    const float* d_ray,
    float* d_tfar,
    const uint* d_matcap,
    int matcapW,
    int matcapH,
    int imageW,
    int imageH
)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    int id = y*imageW + x;

    if (d_mask[id] == 0) return;    // masked out

    int inferenceIndex = d_idSDFMap[id];

    const float3 ray = getFloat3(d_ray, 3*id);
    float3 point = getFloat3(d_points, 3*id);

    if (d_mask[id] >= COLOR_MASK_VAL) {
        // needs to be colored.
        float3 n = surfaceNormal(inferenceIndex, d_sdf, point);
        if (c_coloringType == 0) {
            d_output[id] = facingColor(n, ray);
        } else {
            d_output[id] = matCapColor(n, d_matcap, matcapW, matcapH);
        }
        
        d_mask[id] = 0;
        return;
    }

    const float tstep = sceneSDF(point, d_sdf[inferenceIndex]);

    d_tfar[id] -= tstep;

    if (d_tfar[id] <= 0) {
        d_mask[id] = 0;
        d_output[id] = BACKGROUND_COLOR;
        return;
    }

    // update point along ray
    point = point + ray*tstep;
    setFloat3(d_points, 3*id, point);

    // if close enough, we're done!
    if (tstep < MARCHING_EPSILON) {
        d_mask[id] = COLOR_MASK_VAL;     // prep for coloring!
    };
}

// simple function for debugging 
void printCrap(Image& idSDFMap, Image& stepMask, Matrix& points, Matrix& batch) {
    idSDFMap.copyDeviceToHost();
    stepMask.copyDeviceToHost();
    points.copyDeviceToHost();
    batch.copyDeviceToHost();

    for (int i = 0; i < idSDFMap.size(); i ++) {
        int ptIdx = idSDFMap[i];

        if (stepMask[i] > 0) {
            printf("%d-%d:%d:  (%f,%f,%f)&(%f,%f%f)", i, stepMask[i], idSDFMap[i], points[i*3], points[i*3 + 1], points[i*3 + 2],batch[ptIdx*3], batch[ptIdx*3 + 1], batch[ptIdx*3 + 2]);
            if (stepMask[i] >= COLOR_MASK_VAL) {
                printf(" <---");
            }
            printf("\n");
        }
    }
    printf("BATCH: \n");
    for (int i = 0; i < batch.size()-2; i += 3) {
        printf("\t %d: (%f, %f, %f) \n", i/3, batch[i],batch[i+1],batch[i+2] );
    }
    std::cout << "\n\n";
}

__global__
void createBatch (
    float* d_batch, 
    const uint* d_idSDFMap, 
    const uint* d_mask, 
    const float* d_points, 
    const int imageW, 
    const int imageH) 
{    
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    int idx = y*imageW + x;
    uint maskVal = d_mask[idx];

    if (maskVal == 0) return;

    // index of where to store points in batch
    uint batchIdx = d_idSDFMap[idx]*c_numInputs;    // if frame present. we offset 4 at a time!

    // set points according to mask val
    // mask val == 1 for step request (1 points inference)
    // mask val == 6 for normal request (6 points inference)
    if (maskVal == 1) {
        d_batch[batchIdx] = d_points[idx*3];
        d_batch[batchIdx + 1] = d_points[idx*3 + 1];
        d_batch[batchIdx + 2] = d_points[idx*3 + 2];
        if (c_numInputs == 4) {
            d_batch[batchIdx + 3] = c_frameNumber;
        }
    } else {
        // add all points for normal estimation [x+eps, y, z] [x-eps, y, z] ....
        for (int i = 0; i < 3*maskVal; i += 3) {
            d_batch[batchIdx + i]     = d_points[idx*3]     + tetrahedronVerts[i]*NORMAL_EPSILON;
            d_batch[batchIdx + i + 1] = d_points[idx*3 + 1] + tetrahedronVerts[i+1]*NORMAL_EPSILON;
            d_batch[batchIdx + i + 2] = d_points[idx*3 + 2] + tetrahedronVerts[i+2]*NORMAL_EPSILON;
            if (c_numInputs == 4) {
                d_batch[batchIdx + i + 3] = c_frameNumber;
            }
        }
    }
} 

int formatInferenceReqs(Image& idSDFMap, Image& stepMask, Matrix& points, Matrix& batch, const int imageW, const int imageH, dim3 gridSize, dim3 blockSize) {
    typedef thrust::device_ptr<uint> MatImgPtr;
    
    // d_mask == 1 if step, 6 if normal. required.
    MatImgPtr lastVal = thrust::exclusive_scan(
        thrust::device,
        (MatImgPtr)stepMask.deviceData.get(), 
        (MatImgPtr)(stepMask.deviceData.get() + stepMask.size()), 
        (MatImgPtr)idSDFMap.deviceData.get(),
        0
    );

    int batchSize;
    thrust::copy(lastVal-1, lastVal, &batchSize);
    batch.shape.y = batchSize;

    // prepare batch
    createBatch<<<gridSize, blockSize>>> (
        batch.deviceData.get(),
        idSDFMap.deviceData.get(), 
        stepMask.deviceData.get(), 
        points.deviceData.get(), 
        imageW,
        imageH
    );

    return batchSize;
}

Matrix points;
Matrix batch;
Matrix ray;
Matrix far;
Image stepMask;
Image idSDFMap;

int prevImageSize = -1;

void allocateBuffers(const int imageW, const int imageH, int numInputs) {
    int imageSize = imageW*imageH;

    printf("IMAGE SIZE: %d (%d, %d)\n", imageSize, imageW, imageH);

    printf("ALLOCATING\n");
    points = Matrix(Shape(3, imageSize));
    batch = Matrix(Shape(int(numInputs*COLOR_MASK_VAL), imageSize)); 
    ray = Matrix(Shape(3, imageSize));  // TODO: both of these can easily be dynamic and indexed by idSDFMap
    far = Matrix(Shape(1, imageSize));  // TODO.
    stepMask = Image(Shape(imageW, imageH));
    idSDFMap = Image(Shape(imageW, imageH));

    // allocate them
    points.allocateMemory();
    ray.allocateMemory();
    far.allocateMemory();
    stepMask.allocateMemory();
    idSDFMap.allocateMemory();
    batch.allocateMemory();
    printf("ALLOCATED\n");

    // we need to reinit the sdf buffer
}

extern "C"
void render_kernel(
    dim3 gridSize,
    dim3 blockSize,
    uint *d_output, 
    uint imageW, 
    uint imageH,
    uint numInputs,
    NeuralNetwork& nn,
    Image matcap
) {
    int imageSize = imageH*imageW;

    if (imageSize != prevImageSize) {
        allocateBuffers(imageW, imageH, numInputs);
        prevImageSize = imageSize;
    }
    
    Matrix sdf;

    initMarcher<<<gridSize, blockSize>>> (
        d_output, 
        stepMask.deviceData.get(), 
        points.deviceData.get(), 
        ray.deviceData.get(),
        far.deviceData.get(),
        imageW, 
        imageH, 
        MAX_STEPS
    );

    // remove masked pixel (and reduce batch size if possible)
    int batchSize = formatInferenceReqs(
        idSDFMap,
        stepMask,
        points,
        batch,
        imageW,
        imageH,
        gridSize,
        blockSize
    );

    // march all rays simultaneossly. (so we can utilize batched gemm optimizations)
    for (int i = 0; i < MAX_STEPS; i ++) {   
        
        if (batchSize == 0) {
            //nothing to do!
            break;
        }

        // infer all points required (this has no limit on batch size... large models with large images will exhaust memory very quickly!)
        // TODO: have a toggle to batch our batches into fixed chunks... this would allow us to infer large networks! (second param would be fixed and we chunk data!)
        sdf = nn.forward(batch, int(imageSize*COLOR_MASK_VAL));  // COLOR_MASK_VAL gives number of points required for normal estimation.
        // take step, updating mask, points, and ray position (tfar)
        singleMarch<<<gridSize, blockSize>>>(
            d_output,
            (const uint *)idSDFMap.deviceData.get(), 
            stepMask.deviceData.get(),
            (const float *)sdf.deviceData.get(), 
            points.deviceData.get(), 
            (const float *)ray.deviceData.get(),
            far.deviceData.get(),
            (const uint *)matcap.deviceData.get(),
            matcap.shape.x,
            matcap.shape.y,
            imageW, 
            imageH
        );

        // remove masked pixel (and reduce batch size if possible)
        batchSize = formatInferenceReqs(
            idSDFMap,
            stepMask,
            points,
            batch,
            imageW,
            imageH,
            gridSize,
            blockSize
        );
    }
    //TODO: set any ray that didnt converge to background color
    
}

extern "C"
void copyViewMatrices(float *invViewMatrix, size_t sizeofViewMatrix, float *normalMatrix, size_t sizeofNormalMatrix, int frameNumber, int numInputs)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofViewMatrix));
    checkCudaErrors(cudaMemcpyToSymbol(c_normalMatrix, normalMatrix, sizeofNormalMatrix));
    checkCudaErrors(cudaMemcpyToSymbol(c_frameNumber, &frameNumber, sizeof(int)));
}

extern "C"
void copyStaticSettings(int colorType, int numInputs) {
    checkCudaErrors(cudaMemcpyToSymbol(c_coloringType, &colorType, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(c_numInputs, &numInputs, sizeof(int)));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
