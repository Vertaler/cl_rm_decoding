import functools
import operator as op
import random
import sys
import time

import numpy as np

from utils.ParallelDecoder import ParallelDecoder
from utils.SequentialDecoder import SequentialDecoder
from utils.common import UtilsCommon
from utils.mebius_transform import MebiusTransform


class UtilsError(Exception):
    pass


def ncr(n, r):
    r = min(r, n - r)
    numer = functools.reduce(op.mul, range(n, n - r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def run_with_benchmark(func, *args, **kwargs):
    try:
        start_time = time.time()
        result = func(args, kwargs)
        return result, start_time - time.time()
    except Exception as e:
        print(str(e))
        raise e


def generate_string_function(n, r):
    used_indexes = []
    addends_list = []
    addend_max_count = 0
    # count maximum possible number of addends
    for i in range(1, r + 1):
        addend_max_count += ncr(n, i)

    # generate random addend number
    addend_count = random.choice(range(addend_max_count))

    # generate '0' or '1' constant
    const = random.choice([str(x) for x in range(2)])
    if int(const) == 1:
        addends_list.append(const)

    # generate monom
    for addend_index in range(addend_count):
        while True:
            # regenerate until unique addend is generated
            addend = set(random.sample(range(1, n + 1), random.choice(range(1, r + 1))))
            if addend not in used_indexes:
                break
        used_indexes.append(addend)
        addends_list.append(''.join(['x' + str(x) for x in addend]))

    word = '+'.join(addends_list)

    return word


def generate_error_vector(n, r):
    while True:
        error_vect = random.choice(range(1 << n))
        weight = 0
        for bt in '{:b}'.format(error_vect):
            weight += int(bt)
        if weight <= 2 ** (n - r - 1):
            return np.array(error_vect).astype(np.int8)


def benchmark_rm_decoder(decoder, iters):
    # check decoder for required fields
    if not all([hasattr(decoder, 'n'), hasattr(decoder, 'r'), hasattr(decoder, 'decode')]):
        raise UtilsError('Benchmark error: invalid decoder object')

    # generate test data
    n = decoder.n
    r = decoder.r
    words = []
    errors = []
    for i in range(iters):
        word = generate_error_vector(n, r)
        str_anf_bits = UtilsCommon.bit_form_anf_from_str(word, n)
        vector_anf_bits = list(map(int, str_anf_bits))
        codeword = np.array(vector_anf_bits).astype(np.int8)
        MebiusTransform.exec(codeword, copy=False)
        words.append(codeword)
        error_vector = generate_error_vector(n, r)
        errors.append(error_vector)

    # ren benchmark
    start = time.time()
    for i in range(iters):
        decoder.decode(words[i] ^ errors[i])
    end = time.time()
    return end - start

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python benchmark.py iters n r")
    iters = sys.argv[1]
    n = sys.argv[2]
    r = sys.argv[3]

    par = ParallelDecoder(n,r)
    seq = SequentialDecoder(n,r)

    par_time = benchmark_rm_decoder(par, iters)
    seq_time = benchmark_rm_decoder(seq, iters)
    print(f"Sequential decoder for ({n},{r})-code totaltime: {seq_time} iters: {iters} avgtime: {seq_time/iters}")
    print(f"Parallel decoder for ({n},{r})-code totaltime: {par_time} iters: {iters} avgtime: {par_time/iters}")

