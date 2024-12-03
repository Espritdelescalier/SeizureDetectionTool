from scipy import signal
import pywt
import numpy as np


def pad_data(length, decomp):
    actualDecompLevel = pywt.swt_max_level(length)
    print("Actual decomp level {}".format(actualDecompLevel))
    minLength = length >> actualDecompLevel
    print("Minlength {}".format(minLength))
    while pywt.swt_max_level(minLength) != decomp - actualDecompLevel:
        minLength += 1
        print("Minlength {}".format(minLength))

    print("Remaining levels of decomp {}, resulting decomp of minlength {}".format(decomp-actualDecompLevel,
                                                                                   pywt.swt_max_level(minLength)))
    fullLength = minLength << actualDecompLevel
    pad = (fullLength - length) / 2

    return int(pad)
