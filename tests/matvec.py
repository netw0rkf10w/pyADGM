import numpy as np
import numba
from numba.pycc import CC

cc = CC('matvec')
# Uncomment the following line to print out the compilation steps
#cc.verbose = True

@cc.export('multf', 'f8(f8, f8)')
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    return a * b


@cc.export('dot', 'f8(f8[:], f8[:])')
def dot(x, y):
    """
    compute x.dot(y)
    """
    p = 0.0
    for i in numba.prange(len(x)):
        p += x[i]*y[i]
    return p


@cc.export('eprod', 'f8[:](f8[:], f8[:])')
def eprod(x, y):
    """
    compute element-wise product
    """
    z = np.zeros(x.shape)
    for i in numba.prange(len(x)):
        z[i] = x[i]*y[i]
    return z


# @cc.export('matvec', 'f8[:](f8[:], f8[:], i4[:], i4[:], i4[:])')
# def matvec(x, Adata, Aindices, Aindptr, Ashape):
#     """
#     Fast sparse matrix-vector multiplication
#     https://stackoverflow.com/a/47830250/2131200
#     Note: the first call of this function will be slow
#         because numba needs to initialize.
#     """
#     numRowsA = Ashape[0]    
#     Ax = np.zeros(numRowsA)

#     for i in numba.prange(numRowsA):
#         Ax_i = 0.0        
#         for dataIdx in range(Aindptr[i], Aindptr[i+1]):
#             j = Aindices[dataIdx]
#             Ax_i += Adata[dataIdx]*x[j]

#         Ax[i] = Ax_i
#     return Ax

@cc.export('matvec', 'f8[:](f8[:], f8[:], i4[:], i4[:], i4)')
def matvec(x, Adata, Aindices, Aindptr, numRowsA):
    """
    Fast sparse matrix-vector multiplication
    https://stackoverflow.com/a/47830250/2131200
    Note: the first call of this function will be slow
        because numba needs to initialize.
    """
    Ax = np.zeros(numRowsA)
    for i in numba.prange(numRowsA):
        Ax_i = 0.0        
        for dataIdx in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[dataIdx]
            Ax_i += Adata[dataIdx]*x[j]

        Ax[i] = Ax_i
    return Ax

if __name__ == "__main__":
    cc.compile()