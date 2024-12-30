#include "stdio.h"
#include "stdint.h"
#include "time.h"
#include "assert.h"
#include "immintrin.h"
#include "math.h"

#define N 1024
#define BLOCK 4

float A[N*N] __attribute__ ((aligned(16)));
float B[N*N] __attribute__ ((aligned(16)));
float C[N*N] __attribute__ ((aligned(16)));
float val[N*N] __attribute__ ((aligned(16)));

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
#ifdef FAST
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
            __m256 tc[BLOCK];
            for(int y = 0; y < BLOCK; ++y){
                __m256 tmp = {};
                for(int k = 0; k < N; k +=8){
                    tmp = _mm256_fmadd_ps(
                        Am[((by+y)*N + k)/8],
                        Bm[(bx*N + k)/8],tmp);
                }
                tc[y] = tmp;
            }

            for(int y = 0; y < BLOCK; ++y)
            {
                Cm[((by+y)*N + bx)/8] = tc[y];
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
