#include "stdio.h"
#include "stdint.h"
#include "time.h"
#include "assert.h"
#include "immintrin.h"
#include "math.h"

#define N 2048
#define BLOCK 32

float A[N*N] __attribute__ ((aligned(32)));
float B[N*N] __attribute__ ((aligned(32)));
float C[N*N] __attribute__ ((aligned(32)));
float val[N*N] __attribute__ ((aligned(32)));

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return start.tv_sec*1000000000 + start.tv_nsec;
}

#define FAST

int main()
{
    assert(N%BLOCK == 0);
    FILE *f = fopen("./GEMM_OUT", "rb");
    size_t err;
    err = fread(A, 1, sizeof(float)*N*N, f);
    err = fread(B, 1, sizeof(float)*N*N, f);
    err = fread(val, 1, sizeof(float)*N*N, f);
    fclose(f);

    uint64_t start = nanos();
    printf("finished initializing\r\n");
    for(int by = 0; by < N; by += BLOCK){
        for(int bx = 0; bx < N; bx += BLOCK){
#ifndef FAST
            //compute
            float tc[BLOCK][BLOCK];
            for(int y = 0; y < BLOCK; ++y){
                for(int x = 0; x < BLOCK; ++x){
                    float acc = 0;
                    for( int k = 0; k < N; ++k){
                        acc += A[(by+y)*N + k] * B[(bx+x)*N + k];
                    }
                    tc[y][x] = acc;
                }
            }

            // store
            for(int y = 0; y < BLOCK; ++y){
                for(int x = 0; x < BLOCK; ++x){
                    C[(by+y)*N + bx+x] = tc[y][x];
                }
            }
#else
            float tc[BLOCK][BLOCK];
            for(int y = 0; y < BLOCK; ++y){
                for(int x = 0; x < BLOCK; ++x)
                {
                    __m256 tmp = {};
                    for(int k = 0; k < N; k += 8){
                        tmp = _mm256_fmadd_ps(
                            Am[((by+y)*N + k)/8],
                            Bm[((bx+x)*N + k)/8],
                            tmp);
                    }
                    float ftmp = 0.0;
                    for(int i = 0; i < 8; ++i) ftmp += tmp[i];
                    tc[y][x] = ftmp;
                }
                
            }

            for(int y = 0; y < BLOCK; ++y)
            {
                for(int x = 0; x < BLOCK; ++x)
                {
                    C[(by+y)*N + bx+x] = tc[y][x];
                }
            }

        
#endif
        }
    }
    uint64_t end = nanos();

    double gflop = (2.0*N*N*N)*1e-9;
    double s = (end - start)*1e-9;
    printf("gflops %f\r\n", gflop/s);
    
    for(int k = 0; k < N*N; ++k){
        if(fabsf(C[k] - val[k]) > 1e-3)
        {
            printf("MISMATCH AT %d, %f != %f\r\n", k, C[k],val[k]);
            return -1;
        }
    }
    printf("finished check!\r\n");

    return 0;

}
