import random
import time

import numpy as np
from utils.common import UtilsCommon
from utils.mebius_transform import MebiusTransform

class UtilsError(Exception):
    pass

def run_with_benchmark(func, *args, **kwargs):
    try:
        start_time = time.time()
        result = func(args, kwargs)
        return result, start_time - time.time()
    except Exception as e:
        print(str(e))
        raise e


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
        codeword = np.random.randint(0, 1, size=(2 ** n,), dtype=np.int8)
        MebiusTransform.exec(codeword, copy=False)
        words.append(codeword.astype(np.int8))
        error_vector = UtilsCommon.generate_error_vector(n, r)
        errors.append(error_vector)

    # ren benchmark
    start = time.time()
    for i in range(iters):
        decoder.decode(words[i] ^ errors[i])
    end = time.time()
    return end - start
