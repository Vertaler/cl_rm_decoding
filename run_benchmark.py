import sys

from utils.ParallelDecoder import ParallelDecoder
from utils.CLSequentialDecoder import CLSequentialDecoder
from utils.benchmark import benchmark_rm_decoder

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python benchmark.py iters n r")
        sys.exit(1)
    iters = int(sys.argv[1])
    n = int(sys.argv[2])
    r = int(sys.argv[3])

    par = ParallelDecoder(n,r)
    seq = CLSequentialDecoder(n, r)

    par_time = benchmark_rm_decoder(par, iters)
    seq_time = benchmark_rm_decoder(seq, iters)
    print(f"Sequential decoder for ({n},{r})-code totaltime: {seq_time} iters: {iters} avgtime: {seq_time/iters}")
    print(f"Parallel decoder for ({n},{r})-code totaltime: {par_time} iters: {iters} avgtime: {par_time/iters}")
