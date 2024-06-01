from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise
import numpy as np
from scipy.optimize import minimize_scalar


def dctbpp(Yr, N):
    # Your code here
    x = Yr.shape[0]//N
    total_bits = 0
    
    for i in range(N):
        for j in range(N):
            Ys = Yr[(x*i):(x*(i+1)), (x*j):(x*(j+1))]
            subimg_bits = bpp(Ys) * ((Ys.shape[0])**2)
            total_bits += subimg_bits

    return total_bits

def dct(X, N=8):
    CN = dct_ii(N)
    Y = colxfm(colxfm(X, CN).T, CN).T

    return Y

def idct(Y, N=8):
    CN = dct_ii(N)
    Z = colxfm(colxfm(Y.T, CN.T).T, CN.T)

    return Z


def quantise_selective(Y, N, step_size, rise):
    x = Y.shape[0]//N
    
    for i in range(x):
        for j in range(x):
            start_row = i * N
            end_row = start_row + N
            start_col = j * N
            end_col = start_col + N
            
            if i < 6 and j < 6:
                Y[start_row:end_row, start_col:end_col] = quantise(
                    Y[start_row:end_row, start_col:end_col], step_size, rise * step_size)
            else:
                Y[start_row:end_row, start_col:end_col] = quantise(
                    Y[start_row:end_row, start_col:end_col], step_size, rise * 50 * step_size)

    return Y

def reconstruct_dct(X, step_size, N=8, rise=0.5):
    Xq = quantise(X, 17)
    ref_scheme = bpp(Xq) * (X.shape[0]**2)

    CN = dct_ii(N)
    Y = colxfm(colxfm(X, CN).T, CN).T
    Yq = quantise(Y, step_size, step_size*rise)
    Yr = regroup(Yq, N)
    H_Y_bits = dctbpp(Yr, N)

    dc_ratio = ref_scheme / H_Y_bits
    
    Z = colxfm(colxfm(Yq.T, CN.T).T, CN.T)
    rmse = np.std(X-Z)

    return H_Y_bits, rmse, Z

def reconstruct_dct_suppressed_hf(X, step_size, N=8, rise=0.5):
    Xq = quantise(X, 17)
    ref_scheme = bpp(Xq) * (X.shape[0]**2)

    CN = dct_ii(N)
    Y = colxfm(colxfm(X, CN).T, CN).T
    Yq = quantise_selective(Y, N, step_size, rise)
    Yr = regroup(Yq, N)
    H_Y_bits = dctbpp(Yr, N)

    dc_ratio = ref_scheme / H_Y_bits
    
    Z = colxfm(colxfm(Yq.T, CN.T).T, CN.T)
    rmse = np.std(X-Z)

    return H_Y_bits, rmse, Z

def compression_bits_for_step_size(step_size, X, N, target_bits, rise):
    filter = dct_ii(N)
    Y = colxfm(colxfm(X, filter).T, filter).T
    Yq = quantise(Y, step_size, rise*step_size)
    Yr = regroup(Yq, N)
    bits = dctbpp(Yr, N)
    return abs(bits - target_bits)

def find_step_size_for_compression_bits_dct(X, N, target_bits, rise, min_step_size=0.01, max_step_size=50, tolerance=1e-5):
    result = minimize_scalar(
        compression_bits_for_step_size,
        bounds=(min_step_size, max_step_size),
        args=(X, N, target_bits, rise),
        method='bounded',
        options={'xatol': tolerance}
    )
    return result.x

