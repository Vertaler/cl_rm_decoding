import sys

from decoders.ParallelDecoder import ParallelDecoder
from decoders.SequentialDecoder import SequentialDecoder
from utils.benchmark import benchmark_rm_decoder

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python benchmark.py iters n r")
        print("Use default parameters: 1000 9 3")
        iters = 1000
        n = 9
        r = 2
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
