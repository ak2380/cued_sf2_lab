from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp
from scipy.optimize import minimize_scalar
import numpy as np


def nlevdwt(X, N):
    # your code here
    Y = X.copy()
    m = Y.shape[0]
    L=dwt(Y)
    for _ in range(N-1):
        m = m // 2
        L[:m, :m] = dwt(L[:m, :m])
    return L

def nlevidwt(Y, N):
    # your code here
    m = Y.shape[0]//(2**(N-1))
    Z = Y.copy()
    Z[:m,:m]=idwt(Z[:m,:m])
    for _ in range(N-1):
        m = m * 2
        Z[:m, :m] = idwt(Z[:m, :m])
    return Z

def quantdwt(Y, dwtstep, rise):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here

    N = dwtstep.shape[1] - 1
    Yq = Y.copy()
    dwtent = np.zeros((3, N + 1))

    size = Y.shape[0]
    for i in range(N):
        m = size // (2 ** (i + 1))
        
        for k in range(3):
            if k == 0:  # top right
                sub_image = Y[:m, m:2*m]
                Yq[:m, m:2*m] = quantise(sub_image, dwtstep[k, i], rise*dwtstep[k, i])
                dwtent[k, i] = bpp(Yq[:m, m:2*m])
            elif k == 1:  # bottom left
                sub_image = Y[m:2*m, :m]
                Yq[m:2*m, :m] = quantise(sub_image, dwtstep[k, i], rise*dwtstep[k, i])
                dwtent[k, i] = bpp(Yq[m:2*m, :m])
            elif k == 2:  # bottom right
                sub_image = Y[m:2*m, m:2*m]
                Yq[m:2*m, m:2*m] = quantise(sub_image, dwtstep[k, i], rise*dwtstep[k, i])
                dwtent[k, i] = bpp(Yq[m:2*m, m:2*m])

    m = size // (2 ** (N))
    # Final low-pass image quantization
    final_ll = Y[:m, :m]
    Yq[:m, :m] = quantise(final_ll, dwtstep[0, N], rise*dwtstep[0, N])
    dwtent[0, N] = bpp(Yq[:m, :m])

    return Yq, dwtent

def measure_impulse_response(layer_size, num_levels):
    impulse_responses = np.zeros((3, num_levels + 1))

    for level in range(num_levels):
        Y = np.zeros((layer_size, layer_size))

        sub_size = layer_size // (2 ** (level + 1))
        center = sub_size // 2

        # top right
        Y[sub_size - center, sub_size + center] = 100
        Z_lh = nlevidwt(Y, num_levels)
        impulse_responses[0, level] = np.sum(Z_lh ** 2)
        
        # Reset 
        Y = np.zeros((layer_size, layer_size))
        
        # bottom left
        Y[sub_size + center, sub_size - center] = 100
        Z_hl = nlevidwt(Y, num_levels)
        impulse_responses[1, level] = np.sum(Z_hl ** 2)
        
        # Reset Y
        Y = np.zeros((layer_size, layer_size))
        
        # bottom right
        Y[sub_size + center, sub_size + center] = 100
        Z_hh = nlevidwt(Y, num_levels)
        impulse_responses[2, level] = np.sum(Z_hh ** 2)
        
    # Low-pass image at the final level
    Y = np.zeros((layer_size, layer_size))
    m = layer_size // (2 ** num_levels)
    center = m // 2
    Y[center, center] = 100
    Z_ll = nlevidwt(Y, num_levels)
    impulse_responses[0, num_levels] = np.sum(Z_ll ** 2)
    
    return impulse_responses

def compute_step_sizes(impulse_responses):
    low_pass_energy = impulse_responses[0, -1]
    step_size_ratios = np.zeros_like(impulse_responses)
    
    for level in range(impulse_responses.shape[1]-1):
        for k in range(3):
            step_size_ratios[k, level] = np.sqrt(low_pass_energy / impulse_responses[k, level])
    step_size_ratios[0, impulse_responses.shape[1]-1] = np.sqrt(low_pass_energy / impulse_responses[0, impulse_responses.shape[1]-1])
    
    return step_size_ratios




def compute_H_Yq_bits(dwtent, num_levels, image_size=256):
    H_Yq_bits = 0
    for i in range(num_levels):
        sub_size = (image_size // (2 ** (i+1))) ** 2
        for k in range(3):
            H_Yq_bits += dwtent[k, i] * sub_size
    sub_size = (image_size // (2 ** (num_levels))) ** 2
    H_Yq_bits += dwtent[0, num_levels] * sub_size
    
    return H_Yq_bits

def compute_ss_ratios(N, image_size=256):
    irs = measure_impulse_response(image_size, N)
    ssrs = compute_step_sizes(irs)
    return ssrs


def reconstruct_dwt(X, ssrs, om, N=5, rise=0.5):

    Y = nlevdwt(X, 5)
    dwtstep = ssrs * om
    Yq, dwtent = quantdwt(Y, dwtstep, rise)

    H_Y_bits = compute_H_Yq_bits(dwtent, N)

    Z = nlevidwt(Yq, N)

    rmse = np.std(X - Z)

    return H_Y_bits, rmse, Z


def compression_bits_for_step_size_dwt(om, ssrs, X, N, target_bits, rise):

    Y = nlevdwt(X, N)
    dwtstep = ssrs * om
    Yq, dwtent = quantdwt(Y, dwtstep, rise)
    bits = compute_H_Yq_bits(dwtent, N)
    return abs(bits - target_bits)



def find_step_size_for_compression_bits_dwt(ssrs, X, N, target_bits, rise, min_om=0.01, max_om=50, tolerance=1e-5):
    result = minimize_scalar(
        compression_bits_for_step_size_dwt,
        bounds=(min_om, max_om),
        args=(ssrs, X, N, target_bits, rise),
        method='bounded',
        options={'xatol': tolerance}
    )
    return result.x
