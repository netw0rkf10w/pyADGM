"""
Alternating Direction Graph Matching

Copyright @ D. Khue Le-Huu
https://khue.fr

If you use any part of this code, please cite the following paper.
 
D. Khuê Lê-Huu and Nikos Paragios. Alternating Direction Graph Matching.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

BibTeX:

@inproceedings{lehuu2017adgm,
  title={Alternating Direction Graph Matching},
  author={L{\^e}-Huu, D. Khu{\^e} and Paragios, Nikos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}

This file is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY, without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.sparse

from .utils import simplex_projection_rowwise_, simplex_projection_inequality_rowwise_
from .utils import simplex_projection_sparse, simplex_projection_inequality_sparse


# def rounding(x, u, P):
#     """
#     mat(x): n1 x n2 assignment matrix
#     mat(u): n1 x n2 unary potentials
#     P: n1n2 x n1n2 pairwise potentials
#     """
#     row_ind, col_ind = linear_sum_assignment(distance_matrix)
#     X_new = np.zeros(X.shape, dtype=int)
#     for i, j in zip(row_ind, col_ind):
#         X_new[i, j] = 1
#     return 


def rounding(X, method='hungarian'):
    """
    X: continuous assignment matrix
    method:
        'hungarian': linear assignment rounding
        'greedy': take max first
    """
    if method == 'hungarian':
        row_ind, col_ind = linear_sum_assignment(-X)
        X_disc = np.zeros(X.shape, dtype=int)
        for i, j in zip(row_ind, col_ind):
            X_disc[i, j] = 1
    elif method == 'greedy':
        pass
    else:
        raise NotImplementedError

    return X_disc


def ADGM(U, P, X0=None, **kwargs):
    """
    U: n1 x n2 array representing the unary potentials
    P: (n1n2)x(n1n2) array representing the pairwiste potentials
        P can be a numpy.ndarray or a sparse scipy array ('csr' format)
    """
    rho_min = float(kwargs.get('rho_min', 0.1))
    rho_max = float(kwargs.get('rho_max', 1e5))
    step = float(kwargs.get('step', 1.2))
    precision = float(kwargs.get('precision', 1e-5))
    decrease_delta = float(kwargs.get('decrease_delta', 1e-3))
    iter1 = int(kwargs.get('iter1', 10))
    iter2 = int(kwargs.get('iter2', 20))
    max_iter = int(kwargs.get('max_iter', 10000))
    verbose = bool(kwargs.get('verbose', False))
    both_side_occlusion = bool(kwargs.get('both_side_occlusion', False))
    rounding_method = kwargs.get('rounding_method', 'hungarian')
    # When occlusion is allowed on both side, we remove all matches
    # that has their soft value smaller than this
    rounding_threshold = float(kwargs.get('rounding_threshold', 0.5))

    n1, n2 = U.shape
    n = n1*n2
    assert n == P.shape[0] and P.shape[0] == P.shape[1]
    
    # if 'rho_min' not in kwargs:
    #     rho_min = max(10**(-60.0/np.sqrt(n)), 1e-4)
    # else:
    #     rho_min = float(kwargs.get('rho_min', 0.001))

    # print('rho_min =', rho_min)

    # normalization
    absMax = max(abs(P).max(), abs(U).max())
    rho_min = rho_min*absMax

    u = U.flatten()

    # Initialization
    if X0 is not None:
        x1 = X0.flatten().copy()
    else:
        x1 = np.zeros(n) + 1.0/max(n1, n2)
    x2 = x1.copy()
    y = np.zeros(n)

    rho = rho_min
    iter1_cumulated = iter1
    res_best_so_far = float('inf')
    counter = 0
    residual = float('inf')
    residuals = []

    for k in range(max_iter):
        # Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        # c1 = x2 - (d + M*x2 + y)/rho
        x1_old = x1.copy()
        c1 = x2 - (u + P.dot(x2) + y)/rho

        # Simplex projection for each row of C1
        if both_side_occlusion or n1 > n2:
            x1 = simplex_projection_inequality_rowwise_(c1.reshape((n1, n2))).flatten()
        else:
            x1 = simplex_projection_rowwise_(c1.reshape((n1, n2))).flatten()

        # Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        # c2 = x1 - (M^T*x1 - y)/rho
        x2_old = x2.copy()
        c2 = x1 + (y - P.dot(x1))/rho

        # Simplex projection for each column of C2
        if both_side_occlusion or n2 > n1:
            x2 = simplex_projection_inequality_rowwise_(c2.reshape((n1, n2)).transpose()).transpose().flatten()
        else:
            x2 = simplex_projection_rowwise_(c2.reshape((n1, n2)).transpose()).transpose().flatten()
        

        # Step 3: update y
        y += rho*(x1 - x2)

        # Step 4: compute the residuals and update rho
        r = np.linalg.norm(x1 - x2)
        s = np.linalg.norm(x1 - x1_old) + np.linalg.norm(x2 - x2_old)

        residual_old = residual
        residual = r + s

        # energy = computeEnergy(x1, d, M)
        # energies.push_back(energy)
        residuals.append(residual)

#        if(energy < energy_best){
#            energy_best = energy
#            x = x1
#        }

        if verbose and k%1==0:
            print("%d\t residual = %f"%(k+1, residual))

        # If convergence
        if residual <= precision and residual_old <= precision:
            break

        # Only after iter1 iterations that we start to track the best residuals
        if k >= iter1_cumulated:
            if residual < res_best_so_far - decrease_delta:
                counter = 0
            else:
                counter += 1
            if residual < res_best_so_far:
                res_best_so_far = residual
            
            # If the best_so_far residual has not changed during iter2 iterations, then update rho
            if counter >= iter2:
                if rho < rho_max:
                    rho = min(rho*step, rho_max)
                    if verbose:
                        print('\t UPDATE RHO = %f at iteration %d'%(rho/absMax, k+1))
                    counter = 0
                    iter1_cumulated = k + iter1
                else:
                    break

    if both_side_occlusion:
        raise NotImplementedError
    else:
        X = rounding(x2.reshape((n1, n2)), method=rounding_method)

    # Continuous energy
    if verbose:
        e_continuous = x2.dot(u) + x2.dot(P.dot(x2))
        x = X.flatten()
        e_discrete = x.dot(u) + x.dot(P.dot(x))
        print('Energy: continuous = {}, discrete = {}'.format(e_continuous, e_discrete))
        # print('Assignment matrix =', X1)

    return X



def ADGM_sparse_assignment(U, P, assignment_mask=None, X0=None, **kwargs):
    """
    U: n1 x n2 array representing the unary potentials
    P: A x A array representing the pairwiste potentials
        A = number of non-zeros in assignment_mask if assignment_mask is not None
        A = n1*n2 if assignment_mask is None
    assignment_mask: n1 x n2 array representing the assignment mask
    """
    rho_min = float(kwargs.get('rho_min', 0.1))
    rho_max = float(kwargs.get('rho_max', 1e5))
    step = float(kwargs.get('step', 1.2))
    precision = float(kwargs.get('precision', 1e-5))
    decrease_delta = float(kwargs.get('decrease_delta', 1e-3))
    iter1 = int(kwargs.get('iter1', 10))
    iter2 = int(kwargs.get('iter2', 20))
    max_iter = int(kwargs.get('max_iter', 10000))
    verbose = bool(kwargs.get('verbose', False))
    both_side_occlusion = bool(kwargs.get('both_side_occlusion', False))
    rounding_method = kwargs.get('rounding_method', 'hungarian')
    # When occlusion is allowed on both side, we remove all matches
    # that has their soft value smaller than this
    rounding_threshold = float(kwargs.get('rounding_threshold', 0.5))

    n1, n2 = U.shape
    if assignment_mask is None:
        assignment_mask = np.ones((n1, n2))

    x_linear_indices = np.flatnonzero(assignment_mask)
    A = len(x_linear_indices)
    assert A == P.shape[0] and P.shape[0] == P.shape[1]

    u = U.flatten()[x_linear_indices]

    # We will decompose X into two (sparse) matrices X1 and X2
    # X1 has row constraints and X2 has column ones
    # here we precompute the CSR format for X1 and CSC format for X2
    CSR = scipy.sparse.csr_matrix(assignment_mask)
    CSC = CSR.tocsc()
    
    # For faster conversion of the data array between CSR and CSC format
    # we pre-compute their mappings as well
    CSR.data = np.arange(A) + 1
    csr2csc = scipy.sparse.csc_matrix(CSR.todense()).data - 1
    CSC.data = np.arange(A) + 1
    csc2csr = scipy.sparse.csr_matrix(CSC.todense()).data - 1

    
    # if 'rho_min' not in kwargs:
    #     rho_min = max(10**(-60.0/np.sqrt(n)), 1e-4)
    # else:
    #     rho_min = float(kwargs.get('rho_min', 0.001))

    # print('rho_min =', rho_min)

    # normalization
    absMax = max(abs(P).max(), abs(U).max())
    rho_min = rho_min*absMax

    # the sparse pairwise potential matrix P (shape n x n) is represented by a vector: pPot
    # the corresponding pair indices are stored in a 2d matrix pIndex:
    #   pIndex[pidx] = [a, b] where pPot[pidx] = P[a, b]
    # We can also do the samething for the unary potential and the assignment vector:
    # the sparse unary potential matrix U (shape n1 x n2) is represented by a vector: uPot
    # the corresponding assignment indices are stored in a vector uIndex:
    #   uIndex[uidx] = a where uPot[uidx] = U[i, p] where  p = a % n2, i = int((a - p)/n2)
    # the assignment matrix X (shape n1 x n2) is accordingly represented by a vector: x
    #   x[uidx] = X[i, p] where p = a % n2, i = int((a - p)/n2), a = uIndex[uidx]
    # Note that not all elements in X are present in x.
    # In this version, for simplicity we keep the original representation.



    # each node in a graph has a list of potential matches in the other graph

    # Initialization
    if X0 is not None:
        x1 = X0.flatten().copy()[x_linear_indices]
    else:
        # TODO: uniform initialization for non-zeros in assignment_mask
        x1 = (np.zeros(A) + 1.0/max(n1, n2))
    x2 = x1.copy()
    y = np.zeros(A)

    rho = rho_min
    iter1_cumulated = iter1
    res_best_so_far = float('inf')
    counter = 0
    residual = float('inf')
    residuals = []

    for k in range(max_iter):
        # Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        # c1 = x2 - (d + M*x2 + y)/rho
        x1_old = x1.copy()
        c1 = x2 - (u + P.dot(x2) + y)/rho

        # Simplex projection for each row of C1
        if both_side_occlusion or n1 > n2:
            x1 = simplex_projection_inequality_sparse(c1, CSR.indices, CSR.indptr, CSR.shape[0])
        else:
            x1 = simplex_projection_sparse(c1, CSR.indices, CSR.indptr, CSR.shape[0])

        # Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        # c2 = x1 - (M^T*x1 - y)/rho
        x2_old = x2.copy()
        c2 = x1 + (y - P.dot(x1))/rho

        # Simplex projection for each column of C2
        # Convert it to CSC matrix and to projection column-wise
        c2_csc = c2[csr2csc]
        if both_side_occlusion or n2 > n1:
            x2_csc = simplex_projection_inequality_sparse(c2_csc, CSC.indices, CSC.indptr, CSC.shape[1])
        else:
            x2_csc = simplex_projection_sparse(c2_csc, CSC.indices, CSC.indptr, CSC.shape[1])
        # convert the results back to to CSR matrix
        x2 = x2_csc[csc2csr]

        # Step 3: update y
        y += rho*(x1 - x2)

        # Step 4: compute the residuals and update rho
        r = np.linalg.norm(x1 - x2)
        s = np.linalg.norm(x1 - x1_old) + np.linalg.norm(x2 - x2_old)

        residual_old = residual
        residual = r + s

        # energy = computeEnergy(x1, d, M)
        # energies.push_back(energy)
        residuals.append(residual)

#        if(energy < energy_best){
#            energy_best = energy
#            x = x1
#        }

        if verbose and k%1==0:
            print("%d\t residual = %f"%(k+1, residual))

        # If convergence
        if residual <= precision and residual_old <= precision:
            break

        # Only after iter1 iterations that we start to track the best residuals
        if k >= iter1_cumulated:
            if residual < res_best_so_far - decrease_delta:
                counter = 0
            else:
                counter += 1
            if residual < res_best_so_far:
                res_best_so_far = residual
            
            # If the best_so_far residual has not changed during iter2 iterations, then update rho
            if counter >= iter2:
                if rho < rho_max:
                    rho = min(rho*step, rho_max)
                    if verbose:
                        print('\t UPDATE RHO = %f at iteration %d'%(rho/absMax, k+1))
                    counter = 0
                    iter1_cumulated = k + iter1
                else:
                    break

    if both_side_occlusion:
        raise NotImplementedError
    else:
        # convert x2 to original shape assignment matrix
        CSR.data = x2
        X2 = CSR.todense()
        X = rounding(X2, method=rounding_method)
        X = np.logical_and(X, X2)

    # Continuous energy
    # if verbose:
    #     e_continuous = x2.dot(u) + x2.dot(P.dot(x2))
    #     x = X.flatten()
    #     e_discrete = x.dot(u) + x.dot(P.dot(x))
    #     print('Energy: continuous = {}, discrete = {}'.format(e_continuous, e_discrete))
    #     # print('Assignment matrix =', X1)

    return X



# def ADGM_sparse(U, Pdata, Pindices, X0=None, **kwargs):
#     """
#     U: n1 x n2 array representing the unary potentials
#     P: (n1n2)x(n1n2) array representing the pairwiste potentials
#         P can be a numpy.ndarray or a sparse scipy array ('csr' format)
#     """
#     rho_min = float(kwargs.get('rho_min', 0.1))
#     rho_max = float(kwargs.get('rho_max', 1e5))
#     step = float(kwargs.get('step', 1.2))
#     precision = float(kwargs.get('precision', 1e-5))
#     decrease_delta = float(kwargs.get('decrease_delta', 1e-3))
#     iter1 = int(kwargs.get('iter1', 10))
#     iter2 = int(kwargs.get('iter2', 50))
#     max_iter = int(kwargs.get('max_iter', 10000))
#     verbose = bool(kwargs.get('verbose', False))
#     both_side_occlusion = bool(kwargs.get('both_side_occlusion', False))
#     rounding_method = kwargs.get('rounding_method', 'hungarian')
#     # When occlusion is allowed on both side, we remove all matches
#     # that has their soft value smaller than this
#     rounding_threshold = float(kwargs.get('rounding_threshold', 0.5))

#     n1, n2 = U.shape
#     n = n1*n2
#     assert n == P.shape[0] and P.shape[0] == P.shape[1]
    
#     # if 'rho_min' not in kwargs:
#     #     rho_min = max(10**(-60.0/np.sqrt(n)), 1e-4)
#     # else:
#     #     rho_min = float(kwargs.get('rho_min', 0.001))

#     print('rho_min =', rho_min)

#     # normalization
#     absMax = max(abs(P).max(), abs(U).max())
#     rho_min = rho_min*absMax

#     u = U.flatten()

#     # the sparse pairwise potential matrix P (shape n x n) is represented by a vector: pPot
#     # the corresponding pair indices are stored in a 2d matrix pIndex:
#     #   pIndex[pidx] = [a, b] where pPot[pidx] = P[a, b]
#     # We can also do the samething for the unary potential and the assignment vector:
#     # the sparse unary potential matrix U (shape n1 x n2) is represented by a vector: uPot
#     # the corresponding assignment indices are stored in a vector uIndex:
#     #   uIndex[uidx] = a where uPot[uidx] = U[i, p] where  p = a % n2, i = int((a - p)/n2)
#     # the assignment matrix X (shape n1 x n2) is accordingly represented by a vector: x
#     #   x[uidx] = X[i, p] where p = a % n2, i = int((a - p)/n2), a = uIndex[uidx]
#     # Note that not all elements in X are present in x.
#     # In this version, for simplicity we keep the original representation.



#     # each node in a graph has a list of potential matches in the other graph



#     # Initialization
#     if X0 is not None:
#         x1 = X0.flatten().copy()
#     else:
#         x1 = np.zeros(n) + 1.0/max(n1, n2)
#     x2 = x1.copy()
#     y = np.zeros(n)

#     rho = rho_min
#     iter1_cumulated = iter1
#     res_best_so_far = float('inf')
#     counter = 0
#     residual = float('inf')
#     residuals = []

#     for k in range(max_iter):
#         # Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
#         # c1 = x2 - (d + M*x2 + y)/rho
#         x1_old = x1
#         c1 = matvec_fast(x2, Pdata, Pindices, Pindptr, Pshape)
#         c1 = x2 - (u + c1 + y)/rho

#         # Simplex projection for each row of C1
#         if both_side_occlusion or n1 > n2:
#             x1 = simplex_projection_inequality_rowwise(c1.reshape((n1, n2))).flatten()
#         else:
#             x1 = simplex_projection_rowwise(c1.reshape((n1, n2))).flatten()

#         # Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
#         # c2 = x1 - (M^T*x1 - y)/rho
#         x2_old = x2
#         c2 = matvec_fast(x1, Pdata, Pindices, Pindptr, Pshape)
#         c2 = x1 + (y - c2)/rho

#         # Simplex projection for each row of C1
#         if both_side_occlusion or n2 > n1:
#             x2 = simplex_projection_inequality_rowwise(c2.reshape((n1, n2))).flatten()
#         else:
#             x2 = simplex_projection_rowwise(c2.reshape((n1, n2)).transpose()).transpose().flatten()
        

#         # Step 3: update y
#         y += rho*(x1 - x2)

#         # Step 4: compute the residuals and update rho
#         r = np.linalg.norm(x1 - x2)
#         s = np.linalg.norm(x1 - x1_old) + np.linalg.norm(x2 - x2_old)

#         residual_old = residual
#         residual = r + s

#         # energy = computeEnergy(x1, d, M)
#         # energies.push_back(energy)
#         residuals.append(residual)

# #        if(energy < energy_best){
# #            energy_best = energy
# #            x = x1
# #        }

#         if verbose and k%1==0:
#             print("%d\t residual = %f"%(k+1, residual))

#         # If convergence
#         if residual <= precision and residual_old <= precision:
#             break

#         # Only after iter1 iterations that we start to track the best residuals
#         if k >= iter1_cumulated:
#             if residual < res_best_so_far - decrease_delta:
#                 counter = 0
#             else:
#                 counter += 1
#             if residual < res_best_so_far:
#                 res_best_so_far = residual
            
#             # If the best_so_far residual has not changed during iter2 iterations, then update rho
#             if counter >= iter2:
#                 if rho < rho_max:
#                     rho = min(rho*step, rho_max)
#                     if verbose:
#                         print('\t UPDATE RHO = %f at iteration %d'%(rho/absMax, k+1))
#                     counter = 0
#                     iter1_cumulated = k + iter1
#                 else:
#                     break

#     if both_side_occlusion:
#         raise NotImplementedError
#     else:
#         X = rounding(x2.reshape((n1, n2)), method=rounding_method)

#     # Continuous energy
#     if verbose:
#         e_continuous = x2.dot(u) + x2.dot(P.dot(x2))
#         x = X.flatten()
#         e_discrete = x.dot(u) + x.dot(P.dot(x))
#         print('Energy: continuous = {}, discrete = {}'.format(e_continuous, e_discrete))
#         # print('Assignment matrix =', X1)

#     return X