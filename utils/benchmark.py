import functools
import operator as op
import random
import time

from utils.common import UtilsCommon


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
            return error_vect


def benchmark_rm_decoder(decoder, iters):
    # check decoder for required fields
    if not all([hasattr(decoder, 'n'), hasattr(decoder, 'r'), hasattr(decoder, 'decode')]):
        raise UtilsError('Benchmark error: invalid decoder object')

    # generate test data
    words = []
    errors = []
    for i in range(iters):
        word = generate_error_vector(5, 2)
        byte_word = UtilsCommon.bit_form_anf_from_str(word, 5)
        words.append(byte_word)
        error_vector = generate_error_vector(5, 2)
        errors.append(error_vector)

    # ren benchmark
    start = time.time()
    for i in range(iters):
        decoder.decode(words[i] ^ errors[i])
    end = time.time() - time.time()
