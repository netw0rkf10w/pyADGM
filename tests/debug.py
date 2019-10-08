import numpy as np
import scipy.sparse

def test():
    A = scipy.sparse.random(3, 4, density=0.1, format='csr')
    A.data[:] -= 0.5

    print('data\n', A.data)
    print('indices\n', A.indices)
    print('indptr\n', A.indptr)
    print('shape\n', A.shape)
    
    print('sparse\n', A)
    print('dense\n', A.todense())
    # print('abs(A) =', abs(A))

def toto(a, b):
    a = np.zeros(b.shape)
    a[0] = 15

def test2():
    a = np.ones(5)
    b = np.ones(7)
    toto(a, b)

    print(a)
    print(b)

if __name__ == "__main__":
    test()
