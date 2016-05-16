/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

static const int MAX_OPTIONS = 256;
static const int THREAD_N = 256;

// Determined empirically for G80 GPUs
// Experiment with this value to obtain higher performance on 
// GPUs with more or fewer multiprocessors
static const int MULTIBLOCK_THRESHOLD = 8192;

static float *d_Sum;
static float *h_Sum;

__constant__ float d_S[MAX_OPTIONS]; // allocated large enough for 256 options
__constant__ float d_X[MAX_OPTIONS];
__constant__ float d_T[MAX_OPTIONS];

////////////////////////////////////////////////////////////////////////////////////
// Compute an efficient number of CTAs to use per option for the multiblock
// version (MonteCarloKernel()).  These numbers were determined via experimentation
// on G80 GPUs.  Optimal values may be different on other GPUs.
////////////////////////////////////////////////////////////////////////////////////
unsigned int computeNumCTAs(unsigned int optN)
{
    return (optN < 16) ? 64 : 16;
}

////////////////////////////////////////////////////////////////////////////////////
// Allocate intermediate strage for the Monte Carlo integration
////////////////////////////////////////////////////////////////////////////////////
void initMonteCarloGPU(unsigned int optN, unsigned int pathN, float *h_S, float *h_X, float *h_T){
    
    unsigned int ratio = pathN / optN;
    unsigned int accumSz = 2 * sizeof(float);
    if (ratio >= MULTIBLOCK_THRESHOLD) 
    {
        // in this case we need to store a number of partial sums per thread block
        unsigned int accumN = computeNumCTAs(optN) * THREAD_N;
        accumSz *= accumN;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Use OS-pinned memory on host side. Allocation takes slightly more time,
    // But OS-pinned<==>device memory transfers are faster depending on 
    // the system configuration. Refer to the programming guide and 
    // bandwidthTest CUDA SDK sample for performance comparisons on the
    // particular system.
    ////////////////////////////////////////////////////////////////////////////
    CUDA_SAFE_CALL( cudaMallocHost((void **)&h_Sum,  optN * 2 * sizeof(float)) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Sum,  accumSz * optN) );

    // Initialize the per-option data in constant arrays accessible by MonteCarloKernel()
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_S, h_S, MAX_OPTIONS * sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_X, h_X, MAX_OPTIONS * sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_T, h_T, MAX_OPTIONS * sizeof(float)) );
}

void closeMonteCarloGPU(void){
    CUDA_SAFE_CALL( cudaFree(d_Sum)      );
    CUDA_SAFE_CALL( cudaFreeHost(h_Sum)  );
}

// Needed by the optimized sum reduction for correct execution in device emulation
#ifdef __DEVICE_EMULATION__
#define SYNC __syncthreads()
#else
#define SYNC
#endif

////////////////////////////////////////////////////////////////////////////////////
// Given shared memory with blockSize valus and blockSize squared values,
// This function computes the sum of each array.  The result for each array
// is stored in element 0 of tha array.
////////////////////////////////////////////////////////////////////////////////////
template <unsigned int blockSize>
__device__ void 
sumReduceSharedMem(float *sum, float *sum2)
{
    unsigned int tid = threadIdx.x;

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sum[tid] += sum[tid + 256]; sum2[tid] += sum2[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sum[tid] += sum[tid + 128]; sum2[tid] += sum2[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sum[tid] += sum[tid +  64]; sum2[tid] += sum2[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sum[tid] += sum[tid + 32]; sum2[tid] += sum2[tid + 32]; SYNC; }
        if (blockSize >=  32) { sum[tid] += sum[tid + 16]; sum2[tid] += sum2[tid + 16]; SYNC; }
        if (blockSize >=  16) { sum[tid] += sum[tid +  8]; sum2[tid] += sum2[tid +  8]; SYNC; }
        if (blockSize >=   8) { sum[tid] += sum[tid +  4]; sum2[tid] += sum2[tid +  4]; SYNC; }
        if (blockSize >=   4) { sum[tid] += sum[tid +  2]; sum2[tid] += sum2[tid +  2]; SYNC; }
        if (blockSize >=   2) { sum[tid] += sum[tid +  1]; sum2[tid] += sum2[tid +  1]; SYNC; }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the final sum and sum-of-squares of ACCUM_N values for each option using 
// an optimized parallel tree reduction.  Calls sumReduceSharedMem
////////////////////////////////////////////////////////////////////////////////////
template <unsigned int blockSize>
__global__ void
sumReduction(float *g_odata, float *g_idata, unsigned int blockDataSize)
{
    __shared__ float sum[blockSize];
    __shared__ float sum2[blockSize]; // sum of squares

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*2*blockDataSize + threadIdx.x;
    sum[tid] = 0;
    sum2[tid] = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    for (int count = 0; count < blockDataSize/blockSize; count++)
    {
        sum[tid]  += g_idata[i];  
        sum2[tid] += g_idata[i + blockDataSize];
        i += blockSize;        
    } 
    __syncthreads();

    // do reduction in shared mem
    sumReduceSharedMem<blockSize>(sum, sum2);
    
    // write result for this block to global mem 
    if (tid == 0) 
    {
        g_odata[2 * blockIdx.x]     =  sum[0];
        g_odata[2 * blockIdx.x + 1] = sum2[0];

    }
}

////////////////////////////////////////////////////////////////////////////////////
// This kernel computes partial integrals over all paths using a multiple thread 
// blocks per option.  It is used when a single thread block per option would not
// be enough to keep the GPU busy.  Execution of this kernel is followed by
// a sumReduction() to get the complete integral for each option.
////////////////////////////////////////////////////////////////////////////////////
__global__ void MonteCarloKernel(
    float *d_Sum,    //Partial sums (+sum of squares) destination
    int   accumN,    //Partial sums (sum of squares) count
    float R,         //Risk-free rate
    float V,         //Volatility
    float *d_Random, //N(0, 1) random samples array
    int   pathN      //Sample count
){
    const int tid      = blockDim.x * blockIdx.x + threadIdx.x;
    const int optIndex = blockIdx.y;
    const int threadN  = blockDim.x * gridDim.x;
    float s = d_S[optIndex];
    float x = d_X[optIndex];
    float t = d_T[optIndex];
    
    const float VBySqrtT = V * sqrtf(t);
    const float MuByT    = (R - 0.5f * V * V) * t;

    for(int iAccum = tid; iAccum < accumN; iAccum += threadN){
        float sum = 0, sum2 = 0;

        for(int iPath = iAccum; iPath < pathN; iPath += accumN){
            float              r = d_Random[iPath];
            float  endStockPrice = s * __expf(MuByT + VBySqrtT * r);
            float endOptionPrice = fmaxf(endStockPrice - x, 0);

            sum  += endOptionPrice;
            sum2 += endOptionPrice * endOptionPrice;
        }

        d_Sum[optIndex * 2 * accumN + iAccum +      0] = sum;
        d_Sum[optIndex * 2 * accumN + iAccum + accumN] = sum2;
    }
}

////////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option.  It is fastest when the number of thread blocks times the work per 
// block is high enough to keep the GPU busy.  When this is not the case, using 
// more blocks per option is faster, so we use MonteCarloKernel and sumReduction 
// instead.
////////////////////////////////////////////////////////////////////////////////////
template <unsigned int blockSize>
__global__ void MonteCarloKernelOneBlockPerOption(
    float *d_Sum,    //Partial sums (+sum of squares) destination
    float R,         //Risk-free rate
    float V,         //Volatility
    float *d_Random, //N(0, 1) random samples array
    int   pathN      //Sample count
){
    __shared__ float s_sum[blockSize];
    __shared__ float s_sum2[blockSize]; // sum of squares

    const int tid      = threadIdx.x;
    const int optIndex = blockIdx.y;
    const int threadN  = blockDim.x;
    float s = d_S[optIndex];
    float x = d_X[optIndex];
    float t = d_T[optIndex];
    
    const float VBySqrtT = V * sqrtf(t);
    const float MuByT    = (R - 0.5f * V * V) * t;

    float sum = 0, sum2 = 0;
    for(int iPath = tid; iPath < pathN; iPath += threadN){
        float              r = d_Random[iPath];
        float  endStockPrice = s * __expf(MuByT + VBySqrtT * r);
        float endOptionPrice = fmaxf(endStockPrice - x, 0);

        sum  += endOptionPrice;
        sum2 += endOptionPrice * endOptionPrice;
    }

    s_sum[tid] = sum;
    s_sum2[tid] = sum2;

    __syncthreads();

    // do reduction in shared mem
    sumReduceSharedMem<blockSize>(s_sum, s_sum2);
    
    // write result for this block to global mem 
    if (tid == 0) 
    {
        d_Sum[2 * blockIdx.y]     = s_sum[0];
        d_Sum[2 * blockIdx.y + 1] = s_sum2[0];

    }
}

////////////////////////////////////////////////////////////////////////////////////
// Here we choose between two different methods for performing Monte Carlo 
// integration on the GPU.  When the ratio of paths to options is lower than a 
// threshold (8192 determined empirically for G80 GPUs -- a different threshold is 
// likely to be better on other GPUs!), we run a single kernel that runs one thread 
// block per option and integrates all samples for that option.  This is 
// MonteCarloKernelOneBlockPerOption().  When the ratio is high, then we need more
// threads to better hide memory latency.  In this case we run multiple thread
// blocks per option and compute partial sums stored in the d_Sum array.  This is
// MonteCarloKernel().  These partial sums are then reduced to a final sum using  
// a parallel reduction (sumReduction()).  In both cases, the sum and sum of 
// squares for each option is read back to the host where the final callResult and
// confidenceWidth are computed.  These are computed on the CPU because doing so on
// the GPU would leave most threads idle.
////////////////////////////////////////////////////////////////////////////////////
void MonteCarloGPU(
    float *callResult,      //Call mean expected values array
    float *confidenceWidth, //Call confidence widths array
    float *S,               //Current stock prices array
    float *X,               //Option strike prices array
    float *T,               //Option expiry dates array
    float R,                //Risk-free rates array
    float V,                //Volatilities array
    int optN,               //Input options count
    float *d_Random,        //N(0, 1) random samples array
    int pathN               //Sample count 
){
    int ratio = pathN / optN;

    if (ratio < MULTIBLOCK_THRESHOLD)
    {
        dim3 gridDim(1, optN, 1);
       
        MonteCarloKernelOneBlockPerOption<128><<<gridDim, 128>>>(
            d_Sum, R, V, d_Random, pathN);  
        
        CUT_CHECK_ERROR("MonteCarloKernelBlockPerOption() execution failed\n");
    }
    else
    {
        int ctaN = computeNumCTAs(optN);
        int accumN = ctaN * THREAD_N;
        dim3 gridDim(ctaN, optN, 1);
        MonteCarloKernel<<<gridDim, THREAD_N>>>(
            d_Sum, accumN, R, V, d_Random, pathN);
        CUT_CHECK_ERROR("MonteCarloKernel() execution failed\n");
       
        // Perform a parallel sum reduction on the device to reduce the ACCUM_N values
        // generated per option to a single value (actually two values -- sum and sum of 
        // squares).  This reduction is very efficient on the GPU.
        sumReduction<128><<<optN, 128>>>(d_Sum, d_Sum, accumN);
        CUT_CHECK_ERROR("sumReduction() execution failed\n");
    }
    
    // Read back only the sum and sum of squares for each option to the CPU.
    CUDA_SAFE_CALL( cudaMemcpy(h_Sum, d_Sum, 2 * optN * sizeof(float), cudaMemcpyDeviceToHost) );

    // Compute final statistics
    for(int opt = 0; opt < optN; opt++){
        float sum  = h_Sum[2*opt];
        float sum2 = h_Sum[2*opt+1];

        //Derive average from the total sum and discount by riskfree rate 
        callResult[opt] = (float)(exp(-R * T[opt]) * sum / (double)pathN);

        //Standard deviation
        double stdDev = sqrt(((double)pathN * sum2 - sum * sum)/ ((double)pathN * (double)(pathN - 1)));

        //Confidence width; in 95% of all cases theoretical value lies within these borders
        confidenceWidth[opt] = (float)(exp(-R * T[opt]) * 1.96 * stdDev / sqrt((double)pathN));
    }
}
