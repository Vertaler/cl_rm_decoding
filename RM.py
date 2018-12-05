import sys
from interface.interface import CmdInterface
from utils.common import UtilsCommon as uc
import argparse
from ParallelDecoder import ParallelDecoder
from SequentialDecoder import SequentialDecoder

def config_args_parser(): #parser vars: log versions of vars
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('n', help='n parameter', type=int)
    args_parser.add_argument('r', help='r parameter', type=int)
    args_parser.add_argument('-a', '--anf',    type=str, metavar='\"x1+x2+...\"',
                             help='anf of bits to encode, example: \"x1 + x2 + x2x3\"')
    args_parser.add_argument('-b', '--binary', type=str, metavar='1101...',
                             help='bits to encode, example: 1101')
    args_parser.add_argument('-t', '--action',           metavar='e or d',
                             choices=['e', 'd'],         default='d',
                             help='encode (input as anf of binary) or decode (binary only) (d by defualt)')
    args_parser.add_argument('-m', '--mode',             metavar='p or s',
                             choices=['p', 's'],         default='p',
                             help='working mode (parallel = p or sequential = s). parallel by default (Decode only)',)
    return args_parser

def encode(args):
    #need check: anf or bin
    if args.anf is None and args.binary is None:
        raise AssertionError('Error: nothing to encode')
    iface = CmdInterface()
    iface.read_rm_code_info(args.n, args.r)
    word = args.binary
    if word is None:
        word = uc.bin_form_anf_from_str(args.anf, args.n)
    assert uc.check_string_binary(word), "Error: can't handle non binary word"
    iface.read_rm_code_pure_word(word)
    # TODO encoding here and returnin result
    pass

def decode(args):
    #need to check bin
    if args.binary is None:
        raise AssertionError("Error: nothing to decode")
    assert uc.check_string_binary(args.binary), "Error: can't handle non binary word"
    word = args.binary
    if args.mode == 'p':
        decoder = ParallelDecoder(args.n, args.r)
    else:
        decoder = SequentialDecoder(args.n, args.r)
    result = decoder.decode(uc.np_array_from_bin_str(args.binary))
    return result


if __name__ == "__main__":
    args = sys.argv
    ap = config_args_parser()
    parsed_args = ap.parse_args()
    if len(parsed_args.binary) < 2**parsed_args.n:
        parsed_args.binary += '0' * (2**parsed_args.n - len(parsed_args.binary))
    if parsed_args.action == 'e': #encode
        result = encode(parsed_args)
    else:                         #decode
        result = decode(parsed_args)
        print(result)
    # TODO *тема розовой пантеры*