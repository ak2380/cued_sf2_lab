from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dct import regroup
import numpy as np
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise
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


def pot_forward(X, Pf):
    N = Pf.shape[0]
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

    return Xp

def pot_backward(Zp, Pr):
    N = Pr.shape[0]
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Z = Zp.copy()  #copy the non-transformed edges directly from Z
    Z[:,t] = colxfm(Z[:,t].T, Pr.T).T
    Z[t,:] = colxfm(Z[t,:], Pr.T)
    
    return Z

def lbt_forward(X, N=4, s=1.32):
    Pf, Pr = pot_ii(N,s)
    CN = dct_ii(N)
    Xp = pot_forward(X, Pf)
    Y = colxfm(colxfm(Xp, CN).T, CN).T

    return Y

def lbt_backward(Y, N=4, s=1.32):
    Pf, Pr = pot_ii(N,s)
    CN = dct_ii(N)
    Zp = colxfm(colxfm(Y.T, CN.T).T, CN.T)
    Z = pot_backward(Zp, Pr)

    return Z

def reconstruct_lbt(X, step_size, rise=0.5,N=4, s=1.32):

    Pf, Pr = pot_ii(N,s)
    CN = dct_ii(N)
    Xp = pot_forward(X, Pf)
    Y = colxfm(colxfm(Xp, CN).T, CN).T

    Yq = quantise(Y, step_size, rise*step_size)
    Yr = regroup(Yq, N)

    Zp = colxfm(colxfm(Yq.T, CN.T).T, CN.T)
    Z = pot_backward(Zp, Pr)

    H_Y_bits = dctbpp(Yr, 16)

    rmse = np.std(X-Z)

    return H_Y_bits, rmse, Z

def compression_bits_for_step_size_lbt(step_size, X, N, target_bits, rise):
    Y = lbt_forward(X, N)
    Yq = quantise(Y, step_size, rise*step_size)
    Yr = regroup(Yq, N)
    bits = dctbpp(Yr, 16)
    return abs(bits - target_bits)

def find_step_size_for_compression_bits_lbt(X, N, target_bits, rise, min_step_size=0.01, max_step_size=50, tolerance=1e-5):
    result = minimize_scalar(
        compression_bits_for_step_size_lbt,
        bounds=(min_step_size, max_step_size),
        args=(X, N, target_bits, rise),
        method='bounded',
        options={'xatol': tolerance}
    )
    return result.x