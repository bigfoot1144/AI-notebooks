#include "stdio.h"
#include "stdint.h"
#include "time.h"

#define N 1024

float A[N][N];
float B[N][N];
float C[N][N];

uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return start.tv_sec*1000000000 + start.tv_nsec;
}

int main()
{
    printf("initializing arrays\r\n");
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            A[i][j]= 0;
            B[i][j]= 1;
            C[i][j]= 2;
        }
    }

    uint64_t start = nanos();
    printf("finished initializing\r\n");
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            for(int k = 0; k < N; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    uint64_t end = nanos();

    double tflop = (2.0*N*N*N)*1e-12;
    double s = (end - start)*1e-9;
    printf("tflops %f\r\n", tflop/s);

}