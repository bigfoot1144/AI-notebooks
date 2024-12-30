#include "stdio.h"
#include "stdint.h"
#include "time.h"
#include "assert.h"
#include "immintrin.h"
#include "math.h"

#define N 1024
#define BLOCK 32

float A[N][N];
float B[N][N];
float C[N][N];
float val[N][N];

__m256 *Am = (__m256*)A;

uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return start.tv_sec*1000000000 + start.tv_nsec;
}

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
            //compute
            float tc[BLOCK][BLOCK];
            for(int y = 0; y < BLOCK; ++y){
                for(int x = 0; x < BLOCK; ++x){
                    float acc = 0;
                    for( int k = 0; k < N; ++k){
                        acc += A[by+y][k] * B[bx+x][k];
                    }
                    tc[y][x] = acc;
                }
            }

            // store
            for(int y = 0; y < BLOCK; ++y){
                for(int x = 0; x < BLOCK; ++x){
                    C[by+y][bx+x] = tc[y][x];
                }
            }
            
        }
    }
    uint64_t end = nanos();

    double gflop = (2.0*N*N*N)*1e-9;
    double s = (end - start)*1e-9;
    printf("gflops %f\r\n", gflop/s);
    
    for(int y = 0; y < N; ++y){
        for(int x = 0; x < N; ++x){
            if(fabsf(C[y][x] - val[y][x]) > 1e-3)
            {
                printf("MISMATCH AT %d x %d, %f != %f\r\n", y,x, C[y][x],val[y][x]);
                return -1;
            }
        }
    } 

    return 0;

}
