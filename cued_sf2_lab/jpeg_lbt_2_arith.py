"""
With thanks to 2019 SF2 Group 7 (Jason Li - jl944@cam.ac.uk, Karthik Suresh -
ks800@cam.ac.uk), who did the bulk of the porting from matlab to Python.
"""
from typing import Tuple, NamedTuple, Optional

import numpy as np
from .laplacian_pyramid import quant1, quant2
from .dct import dct_ii, colxfm, regroup, unregroup
from .bitword import bitword
from chosen_schemes.lbt import lbt_forward, lbt_backward

__all__ = [
    "diagscan",
    "runampl",
    "huffdflt",
    "huffgen",
    "huffdes",
    "huffenc",
    "dwtgroup",
    "jpegenc",
    "jpegdec",
    "vlctest",
]


def diagscan(N: int) -> np.ndarray:
    '''
    Generate diagonal scanning pattern

    Returns:
        A diagonal scanning index for a flattened NxN matrix

        The first entry in the matrix is assumed to be the DC coefficient
        and is therefore not included in the scan
    '''
    if N <= 1:
        raise ValueError('Cannot generate a scan pattern for a {}x{} matrix'.format(N, N))

    # Copied from matlab without accounting for indexing.
    slast = N + 1
    scan = [slast]
    while slast != N * N:
        while slast > N and slast % N != 0:
            slast = slast - (N - 1)
            scan.append(slast)
        if slast < N:
            slast = slast + 1
        else:
            slast = slast + N
        scan.append(slast)
        if slast == N * N:
            break
        while slast < (N * N - N + 1) and slast % N != 1:
            slast = slast + (N - 1)
            scan.append(slast)
        if slast == N * N:
            break
        if slast < (N * N - N + 1):
            slast = slast + N
        else:
            slast = slast + 1
        scan.append(slast)
    # Python indexing
    return np.array(scan) - 1


def runampl(a: np.ndarray) -> np.ndarray:
    '''
    Create a run-amplitude encoding from input stream of integers

    Parameters:
        a: array of integers to encode

    Returns:
        ra: (N, 3) array
            ``ra[:, 0]`` gives the runs of zeros between each non-zero value.
            ``ra[:, 1]`` gives the JPEG sizes of the non-zero values (no of
            bits excluding leading zeros).
            ``ra[:, 2]`` gives the values of the JPEG remainder, which
            is normally coded in offset binary.
    '''
    # Check for non integer values in a
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"Arguments to runampl must be integers, got {a.dtype}")
    b = np.where(a != 0)[0]
    if len(b) == 0:
        ra = np.array([[0, 0, 0]])
        return ra

    # List non-zero elements as a column vector
    c = a[b]
    # Generate JPEG size vector ca = floor(log2(abs(c)) + 1)
    ca = np.zeros(c.shape, dtype=np.int64)
    k = 1
    cb = np.abs(c)
    maxc = np.max(cb)

    ka = [1]
    while k <= maxc:
        ca += (cb >= k)
        k = k * 2
        ka.append(k)
    ka = np.array(ka)

    cneg = np.where(c < 0)[0]
    # Changes expression for python indexing
    c[cneg] = c[cneg] + ka[ca[cneg]] - 1
    # appended -1 instead of 0.
    col1 = np.diff(np.concatenate((np.array([-1]), b))) - 1
    ra = np.stack((col1, ca, c), axis=1)
    ra = np.concatenate((ra, np.array([[0, 0, 0]])))
    return ra

step_size_matrix = np.array([
        [16, 11, 10, 16],
        [12, 12, 14, 19],
        [14, 13, 16, 24],
        [14, 17, 22, 29]
    ])

step_size_matrix = step_size_matrix / 10


# step_size_matrix = np.array([
#         [20, 31, 46, 56],
#         [24, 35, 55, 68],
#         [36, 49, 79, 91],
#         [54, 68, 104, 114]
#     ])

# step_size_matrix = step_size_matrix / 10

class ArithmeticCoder:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.bits_to_follow = 0
        self.output_bits = []
        self.input_bits = []
        self.value = 0
        self.bit_index = 0
        self.rescale_factor = 0xFFFFFFFF

    def encode_symbol(self, freqs, symbol):
        range = self.high - self.low + 1
        self.high = self.low + (range * freqs[symbol + 1]) // freqs[-1] - 1
        self.low = self.low + (range * freqs[symbol]) // freqs[-1]

        while True:
            if self.high < 0x80000000:
                self.output_bits.append(0)
                self.output_bits.extend([1] * self.bits_to_follow)
                self.bits_to_follow = 0
            elif self.low >= 0x80000000:
                self.output_bits.append(1)
                self.output_bits.extend([0] * self.bits_to_follow)
                self.bits_to_follow = 0
            elif self.low >= 0x40000000 and self.high < 0xC0000000:
                self.bits_to_follow += 1
                self.low -= 0x40000000
                self.high -= 0x40000000
            else:
                break

            self.low = (self.low * 2) & self.rescale_factor
            self.high = ((self.high * 2) & self.rescale_factor) | 1

    def finish_encoding(self):
        self.bits_to_follow += 1
        if self.low < 0x40000000:
            self.output_bits.append(0)
            self.output_bits.extend([1] * self.bits_to_follow)
        else:
            self.output_bits.append(1)
            self.output_bits.extend([0] * self.bits_to_follow)

    def get_encoded_bits(self):
        return self.output_bits

    def start_decoding(self, input_bits):
        self.input_bits = input_bits
        self.value = 0
        self.bit_index = 0  # Ensure we start from the beginning of the bitstream
        for _ in range(32):
            self.value = (self.value << 1) | int(self.read_bit())

    def read_bit(self):
        if self.bit_index < len(self.input_bits):
            bit = self.input_bits[self.bit_index]
            self.bit_index += 1
            return int(bit)
        return 0

    def decode_symbol(self, freqs):
        range = self.high - self.low + 1
        cum_freq = (((self.value - self.low + 1) * freqs[-1] - 1) // range)
        symbol = 0
        while freqs[symbol + 1] <= cum_freq:
            symbol += 1

        self.high = self.low + (range * freqs[symbol + 1]) // freqs[-1] - 1
        self.low = self.low + (range * freqs[symbol]) // freqs[-1]

        while True:
            if self.high < 0x80000000:
                pass
            elif self.low >= 0x80000000:
                self.low -= 0x80000000
                self.high -= 0x80000000
                self.value -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xC0000000:
                self.low -= 0x40000000
                self.high -= 0x40000000
                self.value -= 0x40000000
            else:
                break

            self.low = (self.low * 2) & self.rescale_factor
            self.high = ((self.high * 2) & self.rescale_factor) | 1
            self.value = (self.value * 2) | int(self.read_bit())

        return symbol

    def bits_required(self):
        return len(self.output_bits)

    def is_finished(self):
        return self.bit_index >= len(self.input_bits)

def calculate_frequency(data):
    freqs = np.zeros(256 + 1, dtype=int)
    for symbol in data:
        freqs[symbol] += 1
    freqs = np.cumsum(freqs)
    return freqs

def flatten_frequency(freqs):
    return freqs.tolist()

def unflatten_frequency(flat_freqs):
    return np.cumsum(np.array(flat_freqs, dtype=int))



def jpegenc_lbt2_arith(X, qstep, fdq=True, N=4, M=16, dcbits=8, log=True):
    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')

    if log:
        print('Forward {} x {} LBT'.format(N, N))
    Y = lbt_forward(X, N, s=1.32)

    Yr = regroup(Y, N)
    lowpass_index = X.shape[0] // N

    if log:
        print('Second {} x {} LBT on low-pass image'.format(N, N))
    Yr[0:lowpass_index, 0:lowpass_index] = lbt_forward(Yr[0:lowpass_index, 0:lowpass_index], N, s=1.32)

    Yur = unregroup(Yr, N)

    if not fdq:
        if log:
            print('Quantising to step size of {}'.format(qstep))
        Yq = quant1(Yur, qstep, qstep).astype('int')

    if fdq:
        if log:
            print('Performing frequency dependent quantisation with overall step size of {}'.format(qstep))
        coeff_table = step_size_matrix
        coeffs = np.tile(coeff_table, (64, 64))
        coeffs = coeffs * qstep

        Yq = np.zeros((256, 256))

        for i in range(256):
            for j in range(256):
                stepsize = coeffs[i, j]
                Yq[i, j] = quant1(Yur[i, j], stepsize, stepsize)
                if Yq[i, j] > 127:
                    Yq[i, j] = 127
        Yq = Yq.astype('int')

    scan = diagscan(M)

    if log:
        print('Coding rows')
    sy = Yq.shape
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r + M, c:c + M]
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            dccoef = yqflat[0] + 2 ** (dcbits - 1)
            if dccoef not in range(2 ** dcbits):
                raise ValueError('DC coefficients too large for desired number of bits')
            vlc.append(dccoef)
            ra1 = runampl(yqflat[scan])
            vlc.extend(ra1.flatten())
            vlc.append(0)  # Ensure End-of-Block (EOB) symbol is appended to each block

    print("Length of VLC after encoding:", len(vlc))  # Debugging line

    # Flatten VLC for frequency calculation
    freqs = calculate_frequency(vlc)

    encoded_bits = []
    encoder = ArithmeticCoder()
    for symbol in vlc:
        encoder.encode_symbol(freqs, symbol)
    encoder.finish_encoding()

    encoded_bits = encoder.get_encoded_bits()
    totalbits = encoder.bits_required()

    # Add frequency table to the encoded bits
    flat_freqs = flatten_frequency(freqs)
    encoded_bits = flat_freqs + encoded_bits

    return encoded_bits, totalbits



def jpegdec_lbt2_arith(encoded_bits, qstep, fdq=True, N=4, M=16, dcbits=8, W=256, H=256, log=True):
    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')

    scan = diagscan(M)

    # Extract frequency table from encoded bits
    flat_freqs = encoded_bits[:257]
    freqs = unflatten_frequency(flat_freqs)
    encoded_bits = encoded_bits[257:]

    decoder = ArithmeticCoder()
    decoder.start_decoding(encoded_bits)
    vlc = []
    while decoder.bit_index < len(encoded_bits):
        symbol = decoder.decode_symbol(freqs)
        vlc.append(symbol)
        if symbol == 0:
            break

    print("Length of VLC during decoding:", len(vlc))  # Debugging line

    eob = 0  # End of block symbol
    i = 0
    Zq = np.zeros((H, W))

    if log:
        print('Decoding rows')
    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M ** 2)
            cf = 0
            if vlc[i] != dcbits:
                raise ValueError('The bits for the DC coefficient do not agree with vlc table')
            yq[cf] = vlc[i] - 2 ** (dcbits - 1)
            i += 1

            while i < len(vlc) and vlc[i] != eob:
                run = vlc[i] >> 4
                si = vlc[i] & 0xF
                cf += run + 1
                ampl = vlc[i + 1]
                thr = 2 ** (si - 1)
                yq[scan[cf - 1]] = ampl - (ampl < thr) * (2 * thr - 1)
                i += 2

            i += 1
            yq = yq.reshape((M, M)).T

            if M > N:
                yq = regroup(yq, M // N)
            Zq[r:r + M, c:c + M] = yq

    if not fdq:
        if log:
            print('Inverse quantising to step size of {}'.format(qstep))
        Zi = quant2(Zq, qstep, qstep)

    if fdq:
        if log:
            print('Inverse FDQ to step size of {}'.format(qstep))
        coeff_table = step_size_matrix
        coeffs = np.tile(coeff_table, (64, 64))
        coeffs = coeffs * qstep

        Zi = np.zeros((256, 256))

        for i in range(256):
            for j in range(256):
                stepsize = coeffs[i, j]
                Zi[i, j] = quant2(Zq[i, j], stepsize, stepsize)

    if log:
        print('Inverting second {} x {} LBT'.format(N, N))
    Zir = regroup(Zi, N)
    lowpass_index = Zi.shape[0] // N
    Zir[0:lowpass_index, 0:lowpass_index] = lbt_backward(Zir[0:lowpass_index, 0:lowpass_index], N, s=1.32)

    Zr = unregroup(Zir, N)

    if log:
        print('Inverse {} x {} LBT\n'.format(N, N))
    Z = lbt_backward(Zr, N, s=1.32)

    return Z






def vlctest(vlc: np.ndarray) -> int:
    """ Test the validity of an array of variable-length codes.

    Returns the total number of bits to code the vlc data. """
    from numpy.lib.recfunctions import (
        structured_to_unstructured, unstructured_to_structured)
    if not np.all(vlc[:,0] >= 0):
        raise ValueError("Code words must be non-negative")
    bitwords = unstructured_to_structured(vlc, dtype=bitword.dtype)
    bitword.verify(bitwords)
    return bitwords['bits'].sum(dtype=np.intp)



def objective_function_lbt2(step, image, target_bits, fdq=True, N=4, M=16, log=False):
    step = max(0, step)  # Ensure the step size is at least 1
    encodedbits, totalbits = jpegenc_lbt2_arith(image, step, fdq=fdq, N=N, M=M, log=log)
    return abs(totalbits - target_bits)
