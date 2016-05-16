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

/*
 * This sample evaluates fair call price for a
 * given set of European options using Monte-Carlo approach.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <cutil.h>
#include "MersenneTwister.h"



////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b){
    return a - a % b;
}

float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



// forward declaration
void shmoo(unsigned int maxNumOptions, unsigned int maxNumPaths);



///////////////////////////////////////////////////////////////////////////////
// CPU gold function prototype
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(
    float& callMean,       //Call mean expected value
    float& callConfidence, //Call confidence width
    float S,               //Current stock price
    float X,               //Option strike price
    float T,               //Option expiry date
    float R,               //Risk-free rate
    float V,               //Volatility
    float *h_Random,       //N(0, 1) random samples array
    int pathN              //Sample count
);

extern "C" void BlackScholes(
    float& CallResult,
    float S, //Option price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
);



///////////////////////////////////////////////////////////////////////////////
// GPU code
///////////////////////////////////////////////////////////////////////////////
#include "MersenneTwister_kernel.cu"
#include "MonteCarlo_kernel.cu"



///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////
//Risk-free rate
const float RISKFREE = 0.06f;
//Volatility rate
const float VOLATILITY = 0.10f;
//Simulation paths (random samples) count 
const int PATH_N = 24000000;
//Number of outputs per generator; align to even for Box-Muller transform
const int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
//Total numbers of sample to generate
const int    RAND_N = MT_RNG_COUNT * N_PER_RNG;

//Reduce problem size to have reasonable emulation time
#ifndef __DEVICE_EMULATION__
const int  OPT_N = 128;
#else
const int  OPT_N = 4;
#endif



//Do Monte-Carlo on CPU?
#define DO_CPU
#undef DO_CPU

//Print Monte-Carlo and Black-Scholes final results?
#define DO_PRINT_RESULTS
#undef DO_PRINT_RESULTS

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    float
        *d_Random;

    float
        *h_Random;

    float
        h_BlackScholes[OPT_N],
        h_CallResultGPU[OPT_N],
        h_ConfidenceWidthGPU[OPT_N],
        S[OPT_N],
        X[OPT_N],
        T[OPT_N];

    double
        delta, ref, reserve, sum_delta, sum_ref, sum_reserve, gpuTime;

    int opt;
    unsigned int hTimer;

    bool bShmoo = cutCheckCmdLineFlag( argc, argv, "shmoo");

    CUT_DEVICE_INIT();
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );

    if (!bShmoo)
        printf("Loading GPU twisters configurations...\n");
    
        const char *dat_path = cutFindFilePath("MersenneTwister.dat", argv[0]);
        initMTGPU(dat_path);

    if (bShmoo) {
        shmoo(MAX_OPTIONS, PATH_N*2);
    }
    else
    {
        printf("Generating random options...\n");
    
            h_Random = (float *)malloc(RAND_N  * sizeof(float));
            CUDA_SAFE_CALL( cudaMalloc((void **)&d_Random, 2*RAND_N  * sizeof(float)) );

            srand(123);
            for(opt = 0; opt < OPT_N; opt++){
                S[opt] = RandFloat(5.0f, 50.f);
                X[opt] = RandFloat(10.0f, 25.0f);
                T[opt] = RandFloat(1.0f, 5.0f);
                BlackScholes(h_BlackScholes[opt], S[opt], X[opt], T[opt], RISKFREE, VOLATILITY);
            }

        printf("Data init done.\n");

        printf("RandomGPU()...\n");
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            CUT_SAFE_CALL( cutResetTimer(hTimer) );
            CUT_SAFE_CALL( cutStartTimer(hTimer) );
            RandomGPU<<<32, 128>>>(d_Random, N_PER_RNG, 777);
            CUT_CHECK_ERROR("RandomGPU() execution failed\n");
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            CUT_SAFE_CALL( cutStopTimer(hTimer) );
            gpuTime = cutGetTimerValue(hTimer);
        printf("Generated samples : %i \n", RAND_N);
        printf("RandomGPU() time  : %f ms\n", gpuTime);
        printf("Samples per second: %E \n", RAND_N / (gpuTime * 0.001));


        printf("BoxMullerGPU()...\n");
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            CUT_SAFE_CALL( cutResetTimer(hTimer) );
            CUT_SAFE_CALL( cutStartTimer(hTimer) );
            BoxMullerGPU<<<32, 128>>>(d_Random, N_PER_RNG);
            CUT_CHECK_ERROR("BoxMullerGPU() execution failed\n");
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            CUT_SAFE_CALL( cutStopTimer(hTimer) );
            gpuTime = cutGetTimerValue(hTimer);
        printf("Transformed samples : %i \n", RAND_N);
        printf("BoxMullerGPU() time : %f ms\n", gpuTime);
        printf("Samples per second  : %E \n", RAND_N / (gpuTime * 0.001));


        printf("GPU Monte-Carlo simulation...\n");
            initMonteCarloGPU(OPT_N, PATH_N, S, X, T);
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            CUT_SAFE_CALL( cutResetTimer(hTimer) );
            CUT_SAFE_CALL( cutStartTimer(hTimer) );
            MonteCarloGPU(
                h_CallResultGPU, h_ConfidenceWidthGPU,
                S, X, T, RISKFREE, VOLATILITY, OPT_N,
                d_Random, PATH_N
            );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            CUT_SAFE_CALL( cutStopTimer(hTimer) );
            closeMonteCarloGPU();
        gpuTime = cutGetTimerValue(hTimer);
        printf("Options count      : %i \n", OPT_N);
        printf("Simulation paths   : %i \n", PATH_N);
        printf("Total GPU time     : %f ms\n", gpuTime);
        printf("Options per second : %E \n", OPT_N / (gpuTime * 0.001));
    
#ifdef DO_CPU
        float h_CallResultCPU, h_ConfidenceWidthCPU;

        printf("Copying random GPU data to CPU.\n");
            CUDA_SAFE_CALL( cudaMemcpy(h_Random, d_Random, RAND_N * sizeof(float), cudaMemcpyDeviceToHost) );

        printf("CPU Monte-Carlo simulation...\n");
        for(opt = 0; opt < OPT_N; opt++){
            MonteCarloCPU(
                h_CallResultCPU, h_ConfidenceWidthCPU,
                S[opt], X[opt], T[opt], RISKFREE, VOLATILITY,
                h_Random, PATH_N
            );

            delta = fabs(h_CallResultCPU - h_CallResultGPU[opt]);
            ref   = h_CallResultCPU;
            printf("CPU: %f / Absolute: %e / Relative: %e\n", ref, delta, delta / ref);
        }
#endif


        sum_delta   = 0;
        sum_ref     = 0;
        sum_reserve = 0;
        for(opt = 0; opt < OPT_N; opt++){
            delta      = fabs(h_BlackScholes[opt] - h_CallResultGPU[opt]);
            ref        = h_BlackScholes[opt];
            reserve    = h_ConfidenceWidthGPU[opt] / delta;
            if(delta > 1e-6) sum_reserve += reserve;
            sum_delta += delta;
            sum_ref   += ref;
#ifdef DO_PRINT_RESULTS
            printf("MC: %f;\t BS: %f\n", h_CallResultGPU[opt], ref);
            printf("Abs: %e;\t Rel: %e;\t Reserve: %f\n", delta, delta / ref, reserve);
#endif
        }
        sum_reserve /= OPT_N;
        printf("L1 norm: %e\n", sum_delta / sum_ref);
        printf("Average reserve: %f\n", sum_reserve);
        printf((sum_reserve > 1.0f) ? "TEST PASSED\n" : "TEST FAILED\n");


        printf("Shutting down...\n");
            CUDA_SAFE_CALL( cudaFree(d_Random) );
            free(h_Random);

        CUT_SAFE_CALL( cutDeleteTimer( hTimer) );
    }

    CUT_EXIT(argc, argv);
}

//Generate a table of comma-separated values (CSV) that can be
//used to generate performance plots across a range of option 
//and path counts.
void shmoo(unsigned int maxNumOptions, unsigned int maxNumPaths)
{
    float
        *d_Random;

    float
        *h_Random;

    float
        *h_CallResultGPU,
        *h_ConfidenceWidthGPU,
        *S,
        *X,
        *T;

    h_CallResultGPU      = (float*)malloc(maxNumOptions * sizeof(float));
    h_ConfidenceWidthGPU = (float*)malloc(maxNumOptions * sizeof(float));
    S                    = (float*)malloc(maxNumOptions * sizeof(float));
    X                    = (float*)malloc(maxNumOptions * sizeof(float));
    T                    = (float*)malloc(maxNumOptions * sizeof(float));

    int maxNPerRNG = iAlignUp(iDivUp(maxNumPaths, MT_RNG_COUNT), 2);
    int maxRandN   = MT_RNG_COUNT * maxNPerRNG;


    unsigned int hTimer;
    float gpuTime;
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );

    h_Random = (float *)malloc(maxRandN  * sizeof(float));
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Random, maxRandN  * sizeof(float)) );

    srand(123);
    for(int opt = 0; opt < maxNumOptions; opt++){
        S[opt] = RandFloat(5.0f, 50.f);
        X[opt] = RandFloat(10.0f, 25.0f);
        T[opt] = RandFloat(1.0f, 5.0f);
    }

    { // shmoo body
        float *randomGPUTime, *boxMullerGPUTime, *monteCarloGPUTime;
        int   *optLabels, *pathLabels;

        int numRows = 0;
        int numCols = 0;
        
        for (int optN = 8; optN <= maxNumOptions; optN *= 2) {
            numCols = 0;
            for (int pathN = MT_RNG_COUNT; pathN <= maxNumPaths; pathN *= 2) {
                numCols++;
            }
            numRows++;
        }

        optLabels         = (int*)  malloc(numRows * sizeof(int));
        pathLabels        = (int*)  malloc(numCols * sizeof(int));
        randomGPUTime     = (float*)malloc(numCols * numRows * sizeof(float));
        boxMullerGPUTime  = (float*)malloc(numCols * numRows * sizeof(float));
        monteCarloGPUTime = (float*)malloc(numCols * numRows * sizeof(float));
       
        // run each kernel once before timing to ensure all code is loaded to GPU
        RandomGPU<<<32, 128>>>(d_Random, 2, 777);
        BoxMullerGPU<<<32, 128>>>(d_Random, 2);
        initMonteCarloGPU(16, MT_RNG_COUNT, S, X, T);
        MonteCarloGPU(h_CallResultGPU, h_ConfidenceWidthGPU,
                      S, X, T, RISKFREE, VOLATILITY, 16,
                      d_Random, MT_RNG_COUNT);
        closeMonteCarloGPU();
        initMonteCarloGPU(16, 1048576, S, X, T);
        MonteCarloGPU(h_CallResultGPU, h_ConfidenceWidthGPU,
                      S, X, T, RISKFREE, VOLATILITY, 16,
                      d_Random, 1048576);
        closeMonteCarloGPU();

        int count = 0;

        for (int o=0, optN = 8; optN <= maxNumOptions; o++, optN *= 2) {
            optLabels[o] = optN;
            for (int p = 0, pathN = MT_RNG_COUNT; pathN <= maxNumPaths; p++, pathN *= 2) {
                
                pathLabels[p] = pathN;

                int NPerRNG = iAlignUp(iDivUp(pathN, MT_RNG_COUNT), 2);

                    CUDA_SAFE_CALL( cudaThreadSynchronize() );
                    CUT_SAFE_CALL( cutResetTimer(hTimer) );
                    CUT_SAFE_CALL( cutStartTimer(hTimer) );
                    RandomGPU<<<32, 128>>>(d_Random, NPerRNG, 777);
                    CUT_CHECK_ERROR("RandomGPU() execution failed\n");
                    CUDA_SAFE_CALL( cudaThreadSynchronize() );
                    CUT_SAFE_CALL( cutStopTimer(hTimer) );
                    gpuTime = cutGetTimerValue(hTimer);
                
                    randomGPUTime[count] = gpuTime;

                    CUDA_SAFE_CALL( cudaThreadSynchronize() );
                    CUT_SAFE_CALL( cutResetTimer(hTimer) );
                    CUT_SAFE_CALL( cutStartTimer(hTimer) );
                    BoxMullerGPU<<<32, 128>>>(d_Random, NPerRNG);
                    CUT_CHECK_ERROR("BoxMullerGPU() execution failed\n");
                    CUDA_SAFE_CALL( cudaThreadSynchronize() );
                    CUT_SAFE_CALL( cutStopTimer(hTimer) );
                    gpuTime = cutGetTimerValue(hTimer);
                    boxMullerGPUTime[count] = gpuTime;

                    
                    initMonteCarloGPU(optN, pathN, S, X, T);
                    CUDA_SAFE_CALL( cudaThreadSynchronize() );
                    CUT_SAFE_CALL( cutResetTimer(hTimer) );
                    CUT_SAFE_CALL( cutStartTimer(hTimer) );
                
                        MonteCarloGPU(
                        h_CallResultGPU, h_ConfidenceWidthGPU,
                        S, X, T, RISKFREE, VOLATILITY, optN,
                        d_Random, pathN);
                
                    CUDA_SAFE_CALL( cudaThreadSynchronize() );
                    CUT_SAFE_CALL( cutStopTimer(hTimer) );
                    closeMonteCarloGPU();
                    gpuTime = cutGetTimerValue(hTimer);                   
                    monteCarloGPUTime[count] = gpuTime;
                    count++;
            }
        }

        float* times[3];
        times[0] = randomGPUTime;
        times[1] = boxMullerGPUTime;
        times[2] = monteCarloGPUTime;
        char* label[] = {"Random Generation GPU Time", "Box Muller GPU Time", "Monte Carlo GPU Time"};

        for (int i = 0; i < 3; i++)
        {
            printf("%s, # paths, \n", label[i]);
            printf("# Options, ");
            for (int pathN = 0; pathN < numCols; ++pathN)
                printf("%d, ", pathLabels[pathN]);
            printf("\n");
       
            for (int optN = 0; optN < numRows; ++optN) {
                printf("%d,", optLabels[optN]);
                for (int pathN = 0; pathN < numCols; ++pathN) {
                    printf("%f, ", times[i][numCols * optN + pathN]);
                }
                printf("\n");
            }
            printf("\n");
        }

        free(optLabels);
        free(pathLabels);
        free(randomGPUTime);
        free(boxMullerGPUTime);
        free(monteCarloGPUTime);
    }

    CUDA_SAFE_CALL( cudaFree(d_Random) );
    free(h_Random);
    free(h_CallResultGPU);
    free(h_ConfidenceWidthGPU);
    free(S);
    free(X);
    free(T);

    CUT_SAFE_CALL( cutDeleteTimer( hTimer) );

}
