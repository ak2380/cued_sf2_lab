import numpy as np


def dct_ii(N):
    """
    Generate the 1D DCT transform matrix of size N.

    Parameters:
    N (int): Size of DCT matrix required

    Returns:
    C (2D np array): 1D DCT transform matrix

    Uses an orthogonal Type-II DCT.
    Y = C * X tranforms N-vector X into Y.
    """
    # initialise C
    C = np.ones((N, N)) / np.sqrt(N)
    # N+1 needed in range to be inclusive of N
    theta = [((a - 0.5) * (np.pi/N)) for a in list(range(1, N+1))]
    # print(theta)
    g = np.sqrt(2/N)
    for i in range(1, N):
        C[i, :] = g * np.cos([a*(i) for a in theta])

    return C


def dct_iv(N):
    """
    Generate the 1D DCT transform matrix of size N.

    Parameters:
    N (int): Size of DCT matrix required

    Returns:
    C (2D np array): 1D DCT transform matrix

    Uses an orthogonal Type-IV DCT.
    Y = C * X tranforms N-vector X into Y.
    """
    # initialise C
    C = np.ones((N, N)) / np.sqrt(N)
    # N+1 needed in range to be inclusive of N
    theta = [((a - 0.5) * (np.pi/N)) for a in list(range(1, N+1))]
    # print(theta)
    g = np.sqrt(2/N)
    for i in range(0, N):
        C[i, :] = g * np.cos([a*(i+0.5) for a in theta])

    return C


def colxfm(X, C):
    """
    Transforms the columns of X using the tranformation in C.

    Parameters:
    X (np.ndarray): Image whose columns are to be transformed
    C (np.ndarray): N-size 1D DCT coefficients obtained using dct_ii(N)
    Returns:
    Y (np.ndarray): Image with transformed columns

    PS: The height of X must be a multiple of the size of C (N).
    """
    N = len(C)
    [m, n] = X.shape

    # catch mismatch in size of X
    if m % N != 0:
        raise ValueError('colxfm error: height of X not multiple of size of C')

    Y = np.zeros((m, n))
    reps = [a for a in range(N)]

    # transform columns of each horizontal stripe of pixels, N*n
    for i in range(0, m, N):
        Y[[i+r for r in reps], :] = np.matmul(C, X[[i+r for r in reps], :])

    return Y


def regroup(X, N):
    """
    Regroup the rows and columns in X.
    Rows/Columns that are N apart in X are adjacent in Y.

    Parameters:
    X (np.ndarray): Image to be regrouped
    N (list): Size of 1D DCT performed (could give int)

    Returns:
    Y (np.ndarray): Regoruped image
    """
    # if N is a 2-element list, N[0] is used for columns and N[1] for rows.
    # if a single value is given, a square matrix is assumed
    if type(N) == int or type(N) == float:
        N = [N, N]

    [m, n] = X.shape

    if (m % N[0] != 0) or (n % N[1] != 0):
        raise ValueError('regroup error: X dimensions not mutiples of N')

    # regroup row and column indices
    m_list = [a for a in range(m)]
    n_list = [a for a in range(n)]
    # need m/N etc as an integer
    tm = np.reshape((np.reshape(m_list, [N[0], int(m/N[0])], order='F').T),
                    [1, m], order='F')
    tn = np.reshape((np.reshape(n_list, [N[1], int(n/N[1])], order='F').T),
                    [1, n], order='F')

    # create a list of indices
    ixgrid = np.ix_(tm[0], tn[0])
    Y = X[ixgrid]

    return Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from .familiarisation import load_mat_img
    from .familiarisation import prep_cmap_array_plt
    from .familiarisation import plot_image

    # code testing dct_iv
    print(dct_iv(8))
    # code for images
    '''
    N = 8
    C8 = dct_ii(N)
    # print(len(C8))
    # print(C8.shape)
    img = 'lighthouse.mat'
    img_info = 'X'
    cmap_info = {'map', 'map2'}
    X, cmaps_dict = load_mat_img(img, img_info, cmap_info)
    # print(X)
    X = X - 128
    # print(X)
    # Y = colxfm(X, C8)
    Y = colxfm(colxfm(X, C8).T, C8).T
    # plot_image(Y)

    cmap_array = cmaps_dict['map']
    cmap_plt = prep_cmap_array_plt(cmap_array, 'map')
    # plot_image(X, cmap_plt='gray')
    # plot_image(Y)
    print(regroup(Y, N)/N)
    # plot_image(regroup(Y, N)/N, cmap_plt)
    '''
    # code to check produced matrices are the same
    '''
    X = np.array([[1,2,3,4,5,6,7,8],
        [10,20,30,40,50,60,70,80],
        [3,6,9,13,15,17,22,32],
        [4,7,88,97,23,45,34,54],
        [1,2,3,4,5,6,7,8],
        [10,20,30,40,50,60,70,80],
        [3,6,9,13,15,17,22,32],
        [4,7,88,97,23,45,34,54]])
    #print(X)
    C4 = dct_ii(4)
    #print(C4)
    Y = colxfm(colxfm(X, C4).T, C4).T
    #print(Y)
    print(regroup(Y,4)/4)
    #plot_image(Y)
    '''