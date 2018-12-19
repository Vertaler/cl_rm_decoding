import sys

from decoders.ParallelDecoder import ParallelDecoder
from decoders.SequentialDecoder import SequentialDecoder
from utils.benchmark import benchmark_rm_decoder

DEFAULT_ITERS = 1000
DEFAULT_N = 9
DEFAULT_R = 2

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python benchmark.py iters n r")
        print(f"Use default parameters: {DEFAULT_ITERS} {DEFAULT_N} {DEFAULT_R}")
        iters = DEFAULT_ITERS
        n = DEFAULT_N
        r = DEFAULT_R
    else:
        iters = int(sys.argv[1])
        n = int(sys.argv[2])
        r = int(sys.argv[3])
    par = ParallelDecoder(n,r)
    seq = SequentialDecoder(n, r)

    par_time = benchmark_rm_decoder(par, iters)
    seq_time = benchmark_rm_decoder(seq, iters)
    print(f"Sequential decoder for ({n},{r})-code totaltime: {seq_time} iters: {iters} avgtime: {seq_time/iters}")
    print(f"Parallel decoder for ({n},{r})-code totaltime: {par_time} iters: {iters} avgtime: {par_time/iters}")
