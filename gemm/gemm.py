#baseline
import time
import numpy as np

N = 1024

if __name__ == "__main__":
    # N^2
    A = np.random.randn(N,N).astype(np.float32)
    # N^2
    B = np.random.randn(N,N).astype(np.float32)

    flop = N*N*2*N
    print(f"{flop/ 1e9} GFLOP")
    
    st = time.monotonic()
    C = A @ B.T
    et = time.monotonic()
    s = et-st

    with open("./GEMM_OUT", "wb") as f:
        f.write(A)
        f.write(B)
        f.write(C)

    print(f"{flop/s * 1e-9} GFLOPS")
