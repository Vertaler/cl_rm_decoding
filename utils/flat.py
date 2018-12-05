from .common import UtilsCommon as uc


class Flat:

    def __init__(self, flat_core, flat_mask, n):
        self.flat_core = flat_core  # some consts for places without *
        self.flat_mask = flat_mask  # 1 - some consts in flat, 0 - *
        self.n = n  # flat length
        self.dim = n - uc.get_weight(flat_mask, n)

    def __getitem__(self, indx):
        # put in places with * (0 in flat_mask) bits from index, in other places put bits from core
        return Flat.merge_by_mask(indx, self.flat_core, self.flat_mask, self.n)

    # where mask has 0 puts bits from first arg, else from second
    @staticmethod
    def merge_by_mask(first, second, mask, n):
        result = 0
        cur_offset_in_first = 0
        cur_offset_in_second = 0
        for cur_offset_in_mask in range(n):
            if ((1 << cur_offset_in_mask) & mask) > 0:  # if we have 1 in mask now, we should set bit from second here
                if ((1 << cur_offset_in_second) & second) > 0:  # we get current bit from second
                    cur_bit_in_second = 1
                    result |= (1 << cur_offset_in_mask)
                else:
                    cur_bit_in_second = 0
                    result &= (~(1 << cur_offset_in_mask))
                cur_offset_in_second += 1
            else:  # we should set bit from first here
                if ((1 << cur_offset_in_first) & first) > 0:  # we get current bit from first
                    cur_bit_in_first = 1
                    result |= (1 << cur_offset_in_mask)
                else:
                    cur_bit_in_first = 0
                    result &= (~(1 << cur_offset_in_mask))
                cur_offset_in_first += 1
        return result

