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
import numba
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import numba

from .lemmas import (
    simplex_projection_rowwise,
    simplex_projection_inequality_rowwise,
    simplex_projection_sparse,
    simplex_projection_inequality_sparse,
    matvec_csr
)

from .energy import build_coo_potentials


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



def ADGM(U, P, assignment_mask=None, X0=None, **kwargs):
    """ Alternating direction graph matching.
    Args:
        - `U` (`numpy.ndarray`): An `n1 x n2` array representing the unary potentials, 
        i.e., the costs of matching individual nodes. For example, `U[i,p]` can be the 
        dissimilarity between the node `i` of `G1` and the node `p` of `G2`. The more 
        similar the two nodes are, the smaller the matching cost should be.
        - `P` (`numpy.ndarray` or `scipy.sparse.csr_matrix`): An `A x A` array 
        representing the pairwise potentials, where `A` is the number of non-zeros 
        of `assignment_mask` (see below), i.e., the total number of match candidates. 
        If no `assignment_mask` is provided, `A = n1*n2`. A match candidate is a pair 
        `(i, p)`, where `i` in `G1` and `p` in `G2`. Such candidate can be represented 
        by an index `a` (`a = 0, 1,..., (A-1)` that corresponds to the `a`-th element 
        of the flatten `assignment matrix` with zero elements removed. 
        We write `a = (i,p)`. For two match candidates `a = (i,p)` and `b = (j,q)`, 
        the pairwise potential `P[a, b]` represents the dissimilarity between the 
        vectors `ij` and `pq`. If `ij` is not an edge of `G1` or if `pq` is not and 
        edge of `G2`, then `P[a, b]` should be zero. Otherwise, `P[a, b]` should be 
        non-positive.
        - `assignment_mask` (`numpy.ndarray`, optional): An `n1 x n2` array representing
        potential match candidates: `assignment_mask[i, p] = 0` means `i` cannot be
        matched to `p`. If you have prior information on this, you should always set this
        matrix as it will make matching much faster and more accurate.
        - `X0` (`numpy.ndarray`, optional): An `n1 x n2` array used for initializing ADGM.
        - `kwargs` (`dict`, optional): ADGM optimization parameters. For example:
        assignment_mask: n1 x n2 array representing the assignment mask
    Return:
        An `n1 x n2` binary assignment matrix
    """
    rho = float(kwargs.get('rho', 0.01))
    rho_max = float(kwargs.get('rho_max', 100))
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
    CSR = csr_matrix(assignment_mask)
    CSC = CSR.tocsc()
    
    # For faster conversion of the data array between CSR and CSC format
    # we pre-compute their mappings as well
    CSR.data = np.arange(A) + 1
    csr2csc = csc_matrix(CSR.todense()).data - 1
    CSC.data = np.arange(A) + 1
    csc2csr = csr_matrix(CSC.todense()).data - 1

    # normalization
    absMax = max(abs(P).max(), abs(U).max())
    rho = rho*absMax

    # Initialization
    if X0 is not None:
        x1 = X0.flatten().copy()[x_linear_indices]
    else:
        # TODO: uniform initialization for non-zeros in assignment_mask
        x1 = (np.zeros(A) + 1.0/max(n1, n2))
    x2 = x1.copy()
    y = np.zeros(A)

    iter1_cumulated = iter1
    res_best_so_far = float('inf')
    counter = 0
    residual = float('inf')

    for k in range(max_iter):
        # Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        # c1 = x2 - (d + 0.5*M*x2 + y)/rho
        x1_old = x1.copy()
        c1 = x2 - (u + 0.5*P.dot(x2) + y)/rho

        # Simplex projection for each row of C1
        if both_side_occlusion or n1 > n2:
            x1 = simplex_projection_inequality_sparse(c1, CSR.indices, CSR.indptr, CSR.shape[0])
        else:
            x1 = simplex_projection_sparse(c1, CSR.indices, CSR.indptr, CSR.shape[0])

        # Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        # c2 = x1 - (0.5*M^T*x1 - y)/rho
        x2_old = x2.copy()
        c2 = x1 + (y - 0.5*P.dot(x1))/rho

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
        # residuals.append(residual)

        # if(energy < energy_best){
        #     energy_best = energy
        #     x = x1
        # }

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


def ADGM_DenseMatches_Old(U, P, X0=None, **kwargs):
    """ Alternating direction graph matching, in which ALL LHS nodes are potential
    matches of a RHS node, and vice versa.
    Args:
        - `U` (`numpy.ndarray`): An `n1 x n2` array representing the unary potentials, 
        i.e., the costs of matching individual nodes. For example, `U[i,p]` can be the 
        dissimilarity between the node `i` of `G1` and the node `p` of `G2`. The more 
        similar the two nodes are, the smaller the matching cost should be.
        - `P` (`numpy.ndarray` or `scipy.sparse.csr_matrix`): An `A x A` array 
        representing the pairwise potentials, where `A` is the number of non-zeros 
        of `assignment_mask` (see below), i.e., the total number of match candidates. 
        If no `assignment_mask` is provided, `A = n1*n2`. A match candidate is a pair 
        `(i, p)`, where `i` in `G1` and `p` in `G2`. Such candidate can be represented 
        by an index `a` (`a = 0, 1,..., (A-1)` that corresponds to the `a`-th element 
        of the flatten `assignment matrix` with zero elements removed. 
        We write `a = (i,p)`. For two match candidates `a = (i,p)` and `b = (j,q)`, 
        the pairwise potential `P[a, b]` represents the dissimilarity between the 
        vectors `ij` and `pq`. If `ij` is not an edge of `G1` or if `pq` is not and 
        edge of `G2`, then `P[a, b]` should be zero. Otherwise, `P[a, b]` should be 
        non-positive.
        - `X0` (`numpy.ndarray`, optional): An `n1 x n2` array used for initializing ADGM.
        - `kwargs` (`dict`, optional): ADGM optimization parameters. For example:
        assignment_mask: n1 x n2 array representing the assignment mask
    Return:
        An `n1 x n2` binary assignment matrix
    """
    rho = float(kwargs.get('rho', 0.01))
    rho_max = float(kwargs.get('rho_max', 10000))
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
    A = n1*n2
    assert A == P.shape[0] and P.shape[0] == P.shape[1]

    # row_ind, col_ind = linear_sum_assignment(U)
    # X = np.zeros(U.shape)
    # for i, j in zip(row_ind, col_ind):
    #     X[i, j] = 1
    # print(f'X:\n{X}')
    # return X

    u = U.flatten()

    # We will decompose X into two (sparse) matrices X1 and X2
    # X1 has row constraints and X2 has column ones
    # here we precompute the CSR format for X1 and CSC format for X2

    # normalization
    absMax = max(abs(P).max(), abs(U).max())
    rho = rho*absMax
    # print(f'scaled rho = {rho}')

    # Initialization
    if X0 is not None:
        x1 = X0.flatten()
    else:
        # TODO: uniform initialization for non-zeros in assignment_mask
        x1 = np.zeros(A) + 1.0/n2
        x2 = np.zeros(A) + 1.0/n1
        # row_ind, col_ind = linear_sum_assignment(U)
        # X = np.zeros_like(U)
        # for i, j in zip(row_ind, col_ind):
        #     X[i, j] = 1
        # x1 = X.flatten()
        # x2 = X.flatten()
    
    y = np.zeros(A)

    iter1_cumulated = iter1
    res_best_so_far = float('inf')
    counter = 0
    residual = float('inf')

    for k in range(max_iter):
        # Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        # c1 = x2 - (d + M*x2 + y)/rho
        x1_old = x1.copy()
        # c1 = x2 - (u +  P.dot(x2) + y)/rho
        c1 = x2 - (u + matvec_csr(x2, P.data, P.indices, P.indptr, P.shape) + y)/rho

        # Simplex projection for each row of C1
        if both_side_occlusion or n1 > n2:
            x1 = simplex_projection_inequality_rowwise(c1.reshape(n1, n2)).flatten()
        else:
            x1 = simplex_projection_rowwise(c1.reshape(n1, n2)).flatten()

        # Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        # c2 = x1 - (M^T*x1 - y)/rho
        x2_old = x2.copy()
        # c2 = x1 + (y - P.dot(x1))/rho
        c2 = x1 + (y - matvec_csr(x1, P.data, P.indices, P.indptr, P.shape))/rho

        # Simplex projection for each column of C2
        C2 = c2.reshape(n1, n2)
        if both_side_occlusion or n2 > n1:
            x2 = simplex_projection_inequality_rowwise(C2.T).T.flatten()
        else:
            x2 = simplex_projection_rowwise(C2.T).T.flatten()

        # Step 3: update y
        y += rho*(x1 - x2)

        # Step 4: compute the residuals and update rho
        r = np.linalg.norm(x1 - x2)
        s = np.linalg.norm(x1 - x1_old) + np.linalg.norm(x2 - x2_old)

        residual_old = residual
        residual = r + s

        # energy = computeEnergy(x1, d, M)
        # energies.push_back(energy)
        # residuals.append(residual)

        # if(energy < energy_best){
        #     energy_best = energy
        #     x = x1
        # }

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
        X2 = x2.reshape(n1, n2)
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


def ADGM_DenseMatches(U, P, X0=None, **kwargs):
    """ Alternating direction graph matching, in which ALL LHS nodes are potential
    matches of a RHS node, and vice versa.
    Args:
        - `U` (`numpy.ndarray`): An `n1 x n2` array representing the unary potentials, 
        i.e., the costs of matching individual nodes. For example, `U[i,p]` can be the 
        dissimilarity between the node `i` of `G1` and the node `p` of `G2`. The more 
        similar the two nodes are, the smaller the matching cost should be.
        - `P` (`numpy.ndarray` or `scipy.sparse.csr_matrix`): An `A x A` array 
        representing the pairwise potentials, where `A` is the number of non-zeros 
        of `assignment_mask` (see below), i.e., the total number of match candidates. 
        If no `assignment_mask` is provided, `A = n1*n2`. A match candidate is a pair 
        `(i, p)`, where `i` in `G1` and `p` in `G2`. Such candidate can be represented 
        by an index `a` (`a = 0, 1,..., (A-1)` that corresponds to the `a`-th element 
        of the flatten `assignment matrix` with zero elements removed. 
        We write `a = (i,p)`. For two match candidates `a = (i,p)` and `b = (j,q)`, 
        the pairwise potential `P[a, b]` represents the dissimilarity between the 
        vectors `ij` and `pq`. If `ij` is not an edge of `G1` or if `pq` is not and 
        edge of `G2`, then `P[a, b]` should be zero. Otherwise, `P[a, b]` should be 
        non-positive.
        - `X0` (`numpy.ndarray`, optional): An `n1 x n2` array used for initializing ADGM.
        - `kwargs` (`dict`, optional): ADGM optimization parameters. For example:
        assignment_mask: n1 x n2 array representing the assignment mask
    Return:
        An `n1 x n2` binary assignment matrix
    """
    rho = float(kwargs.get('rho', 0.01))
    rho_max = float(kwargs.get('rho_max', 10000))
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
    projection = kwargs.get('projection', 'euclidean')

    n1, n2 = U.shape
    A = n1*n2
    assert A == P.shape[0] and P.shape[0] == P.shape[1]

    # row_ind, col_ind = linear_sum_assignment(U)
    # X = np.zeros(U.shape)
    # for i, j in zip(row_ind, col_ind):
    #     X[i, j] = 1
    # print(f'X:\n{X}')
    # return X

    # We will decompose X into two (sparse) matrices X1 and X2
    # X1 has row constraints and X2 has column ones
    # here we precompute the CSR format for X1 and CSC format for X2

    # normalization
    absMax = max(abs(P).max(), abs(U).max())
    rho = rho*absMax
    # print(f'scaled rho = {rho}')

    # Initialization
    if X0 is None:
        # TODO: uniform initialization for non-zeros in assignment_mask
        X = np.zeros_like(U) + 1.0/n2
        Z = np.zeros_like(U) + 1.0/n1
        # row_ind, col_ind = linear_sum_assignment(U)
        # X = np.zeros_like(U)
        # for i, j in zip(row_ind, col_ind):
        #     X[i, j] = 1
        # Z = X
    
    Y = np.zeros_like(U)

    iter1_cumulated = iter1
    res_best_so_far = float('inf')
    counter = 0
    residual = float('inf')

    for k in range(max_iter):
        # Step 1: update x
        # v = -(u + 0.5*Pz + y)/rho
        X_old = X.copy()
        V = matvec_csr(Z.flatten(), P.data, P.indices, P.indptr, P.shape).reshape(n1, n2)
        if verbose:
            energy = np.sum(U*Z) + 0.5*np.sum(Z*V)
        V = -(U + 0.5*V + Y)/rho

        if projection == 'euclidean':
            # Simplex projection for each row of C1
            if both_side_occlusion or n1 > n2:
                X = simplex_projection_inequality_rowwise(Z + V)
            else:
                X = simplex_projection_rowwise(Z + V)
        elif projection == 'bregman':
            if both_side_occlusion or n1 > n2:
                raise NotImplementedError
            else:
                X = (Z + 1e-10)*np.exp(V - np.max(V, axis=-1, keepdims=True))
                X = X / np.sum(X, axis=-1, keepdims=True)
        else:
            raise NotImplementedError

        # Step 2: update z
        # w = -(0.5*Px - y)/rho
        Z_old = Z.copy()
        W = matvec_csr(X.flatten(), P.data, P.indices, P.indptr, P.shape).reshape(n1, n2)
        W = (Y - 0.5*W)/rho

        if projection == 'euclidean':
            # Simplex projection for each row of C1
            if both_side_occlusion or n2 > n1:
                Z = simplex_projection_inequality_rowwise((X + W).T).T
            else:
                Z = simplex_projection_rowwise((X + W).T).T
        elif projection == 'bregman':
            if both_side_occlusion or n2 > n1:
                raise NotImplementedError
            else:
                Z = (X + 1e-10)*np.exp(W - np.max(W, axis=-2, keepdims=True))
                Z = Z / np.sum(Z, axis=-2, keepdims=True)
        else:
            raise NotImplementedError

        # Step 3: update y
        Y += rho*(X - Z)

        # Step 4: compute the residuals and update rho
        r = np.linalg.norm(X - Z)
        s = np.linalg.norm(X - X_old) + np.linalg.norm(Z - Z_old)
        residual_old = residual
        residual = r + s

        # energy = computeEnergy(x1, d, M)
        # energies.push_back(energy)
        # residuals.append(residual)

        # if(energy < energy_best){
        #     energy_best = energy
        #     x = x1
        # }

        if verbose and k%1==0:
            print(f"{k + 1}\t E = {energy:.7f} res = {residual:.7f}, r = {r:.7f}, s = {s:.7f}")

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

    # if verbose:
    #     print(f'Final Z:\n{Z}')

    if both_side_occlusion:
        raise NotImplementedError
    else:
        X = rounding(Z, method=rounding_method)
        X = np.logical_and(X, Z)

    # Continuous energy
    # if verbose:
    #     e_continuous = x2.dot(u) + x2.dot(P.dot(x2))
    #     x = X.flatten()
    #     e_discrete = x.dot(u) + x.dot(P.dot(x))
    #     print('Energy: continuous = {}, discrete = {}'.format(e_continuous, e_discrete))
    #     # print('Assignment matrix =', X1)

    return X


# @numba.jit(nopython=True, fastmath=True, cache=True)
def ADGM_DenseMatches_Fixed(U, Pdata, Pindices, Pindptr, Pshape, rho, steps):
    """ Alternating direction graph matching, in which ALL LHS nodes are potential
    matches of a RHS node, and vice versa.
    Args:
        - `U` (`numpy.ndarray`): An `n1 x n2` array representing the unary potentials, 
        i.e., the costs of matching individual nodes. For example, `U[i,p]` can be the 
        dissimilarity between the node `i` of `G1` and the node `p` of `G2`. The more 
        similar the two nodes are, the smaller the matching cost should be.
        - `P` (`numpy.ndarray` or `scipy.sparse.csr_matrix`): An `A x A` array 
        representing the pairwise potentials, where `A` is the number of non-zeros 
        of `assignment_mask` (see below), i.e., the total number of match candidates. 
        If no `assignment_mask` is provided, `A = n1*n2`. A match candidate is a pair 
        `(i, p)`, where `i` in `G1` and `p` in `G2`. Such candidate can be represented 
        by an index `a` (`a = 0, 1,..., (A-1)` that corresponds to the `a`-th element 
        of the flatten `assignment matrix` with zero elements removed. 
        We write `a = (i,p)`. For two match candidates `a = (i,p)` and `b = (j,q)`, 
        the pairwise potential `P[a, b]` represents the dissimilarity between the 
        vectors `ij` and `pq`. If `ij` is not an edge of `G1` or if `pq` is not and 
        edge of `G2`, then `P[a, b]` should be zero. Otherwise, `P[a, b]` should be 
        non-positive.
        - `X0` (`numpy.ndarray`, optional): An `n1 x n2` array used for initializing ADGM.
        - `kwargs` (`dict`, optional): ADGM optimization parameters. For example:
        assignment_mask: n1 x n2 array representing the assignment mask
    Return:
        An `n1 x n2` binary assignment matrix
    """
    n1 = U.shape[0]
    n2 = U.shape[1]
    A = n1*n2

    # row_ind, col_ind = linear_sum_assignment(U)
    # X = np.zeros(U.shape)
    # for i, j in zip(row_ind, col_ind):
    #     X[i, j] = 1
    # print(f'X:\n{X}')
    # return X

    u = U.flatten()

    # We will decompose X into two (sparse) matrices X1 and X2
    # X1 has row constraints and X2 has column ones
    # here we precompute the CSR format for X1 and CSC format for X2

    # normalization
    absMax = np.maximum(np.abs(Pdata).max(), np.abs(U).max())
    rho = rho*absMax
    
    # TODO: uniform initialization for non-zeros in assignment_mask
    x1 = np.zeros(A) + 1.0/n2
    x2 = np.zeros(A) + 1.0/n1
    # row_ind, col_ind = linear_sum_assignment(U)
    # X = np.zeros_like(U)
    # for i, j in zip(row_ind, col_ind):
    #     X[i, j] = 1
    # x1 = X.flatten()
    # x2 = X.flatten()
    
    y = np.zeros(A)

    for k in range(steps):
        # Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        # c1 = x2 - (d + M*x2 + y)/rho
        x1_old = x1.copy()
        # c1 = x2 - (u +  P.dot(x2) + y)/rho
        c1 = x2 - (u + matvec_csr(x2, Pdata, Pindices, Pindptr, Pshape) + y)/rho

        # Simplex projection for each row of C1
        if n1 > n2:
            x1 = simplex_projection_inequality_rowwise(c1.reshape(n1, n2)).flatten()
        else:
            x1 = simplex_projection_rowwise(c1.reshape(n1, n2)).flatten()

        # Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        # c2 = x1 - (M^T*x1 - y)/rho
        x2_old = x2.copy()
        # c2 = x1 + (y - P.dot(x1))/rho
        c2 = x1 + (y - matvec_csr(x1, Pdata, Pindices, Pindptr, Pshape))/rho

        # Simplex projection for each column of C2
        C2 = c2.reshape(n1, n2)
        if n2 > n1:
            x2 = simplex_projection_inequality_rowwise(C2.T).T.flatten()
        else:
            x2 = simplex_projection_rowwise(C2.T).T.flatten()

        # Step 3: update y
        y += rho*(x1 - x2)

        # Step 4: compute the residuals and update rho
        r = np.linalg.norm(x1 - x2)
        s = np.linalg.norm(x1 - x1_old) + np.linalg.norm(x2 - x2_old)
        residual = r + s
        print(f"{k + 1}\t residual = {residual}, r = {r}, s = {s}")
    
    # convert x2 to original shape assignment matrix
    # X2 = x2.reshape(n1, n2)
    # X = rounding(X2, method=rounding_method)
    # X = np.logical_and(X, X2)

    # return X

    return x2.reshape(n1, n2)


# @numba.jit(nopython=True, fastmath=True, cache=True)
def ADGM_DenseMatches_Fixed_Bregman(U, Pdata, Pindices, Pindptr, Pshape, rho, steps):
    """ Alternating direction graph matching, in which ALL LHS nodes are potential
    matches of a RHS node, and vice versa.
    Args:
        - `U` (`numpy.ndarray`): An `n1 x n2` array representing the unary potentials, 
        i.e., the costs of matching individual nodes. For example, `U[i,p]` can be the 
        dissimilarity between the node `i` of `G1` and the node `p` of `G2`. The more 
        similar the two nodes are, the smaller the matching cost should be.
        - `P` (`numpy.ndarray` or `scipy.sparse.csr_matrix`): An `A x A` array 
        representing the pairwise potentials, where `A` is the number of non-zeros 
        of `assignment_mask` (see below), i.e., the total number of match candidates. 
        If no `assignment_mask` is provided, `A = n1*n2`. A match candidate is a pair 
        `(i, p)`, where `i` in `G1` and `p` in `G2`. Such candidate can be represented 
        by an index `a` (`a = 0, 1,..., (A-1)` that corresponds to the `a`-th element 
        of the flatten `assignment matrix` with zero elements removed. 
        We write `a = (i,p)`. For two match candidates `a = (i,p)` and `b = (j,q)`, 
        the pairwise potential `P[a, b]` represents the dissimilarity between the 
        vectors `ij` and `pq`. If `ij` is not an edge of `G1` or if `pq` is not and 
        edge of `G2`, then `P[a, b]` should be zero. Otherwise, `P[a, b]` should be 
        non-positive.
        - `X0` (`numpy.ndarray`, optional): An `n1 x n2` array used for initializing ADGM.
        - `kwargs` (`dict`, optional): ADGM optimization parameters. For example:
        assignment_mask: n1 x n2 array representing the assignment mask
    Return:
        An `n1 x n2` binary assignment matrix
    """
    # verbose = True
    # precision = 1e-4

    n1 = U.shape[0]
    n2 = U.shape[1]

    # row_ind, col_ind = linear_sum_assignment(U)
    # X = np.zeros(U.shape)
    # for i, j in zip(row_ind, col_ind):
    #     X[i, j] = 1
    # print(f'X:\n{X}')
    # return X

    # We will decompose X into two (sparse) matrices X1 and X2
    # X1 has row constraints and X2 has column ones
    # here we precompute the CSR format for X1 and CSC format for X2

    # normalization
    absMax = np.maximum(np.abs(Pdata).max(), np.abs(U).max())
    rho = rho*absMax
    
    # TODO: uniform initialization for non-zeros in assignment_mask
    X = np.zeros_like(U) + 1.0/n2
    Z = np.zeros_like(U) + 1.0/n1
    # row_ind, col_ind = linear_sum_assignment(U)
    # X = np.zeros_like(U)
    # for i, j in zip(row_ind, col_ind):
    #     X[i, j] = 1
    # Z = X
    
    Y = np.zeros_like(U)

    for k in range(steps):
        # Step 1: update x
        # v = -(u + 0.5*Pz + y)/rho
        X_old = X.copy()
        V = matvec_csr(Z.flatten(), Pdata, Pindices, Pindptr, Pshape).reshape(n1, n2)
        # if verbose:
        #     energy = np.sum(U*Z) + 0.5*np.sum(Z*V)
        V = -(U + 0.5*V + Y)/rho

        # Simplex projection for each row of C1
        if n1 > n2:
            # X = (Z + 1e-10)*np.exp(V - np.max(V, axis=-1, keepdims=True))
            # ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
            X = (Z + 1e-10)*np.exp(V - V.max(-1, None, True))
            X = X / np.sum(X, axis=-1, keepdims=True)
        else:
            # X = (Z + 1e-10)*np.exp(V - np.max(V, axis=-1, keepdims=True))
            # ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
            X = (Z + 1e-10)*np.exp(V - V.max(-1, None, True))
            X = X / np.sum(X, axis=-1, keepdims=True)
        # print(f'X normalized sum over row:\n {np.sum(X, axis=-1)}')

        # Step 2: update z
        # w = -(0.5*Px - y)/rho
        Z_old = Z.copy()
        W = matvec_csr(X.flatten(), Pdata, Pindices, Pindptr, Pshape).reshape(n1, n2)
        W = (Y - 0.5*W)/rho

        # Simplex projection for each column of C2
        if n2 > n1:
            # Z = (X + 1e-10)*np.exp(W - np.max(W, axis=-2, keepdims=True))
            # ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
            Z = (X + 1e-10)*np.exp(W - W.max(-2, None, True))
            Z = Z / np.sum(Z, axis=-2, keepdims=True)
        else:
            # Z = (X + 1e-10)*np.exp(W - np.max(W, axis=-2, keepdims=True))
            # ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
            Z = (X + 1e-10)*np.exp(W - W.max(-2, None, True))
            Z = Z / np.sum(Z, axis=-2, keepdims=True)

        # Step 3: update y
        Y += rho*(X - Z)

        # Step 4: compute the residuals and update rho
        # if precision > 0:
        #     r = np.linalg.norm(X - Z)
        #     s = np.linalg.norm(X - X_old) + np.linalg.norm(Z - Z_old)
        #     residual = r + s
        #     # if verbose:
        #     #     print(f"{k + 1}\t residual = {residual:.7f}, r = {r:.7f}, s = {s:.7f}")
        #     if residual < precision:
        #         break

    return Z


@numba.jit(nopython=True, parallel=True, cache=True)
def build_XQ(X, edges1, edges2):
    """Compute the quadratic part of the assignment vector x in linear programming
    relaxation
    """
    E1 = edges1.shape[0]
    E2 = edges2.shape[0]
    XQ = np.zeros((E1, E2))

    for idx in numba.prange(E1*E2):
        e1 = idx // E2
        e2 = idx % E2
        i, j = edges1[e1]
        p, q = edges2[e2]
        XQ[e1, e2] = X[i, p]*X[j, q]

    return XQ


def ADGM_D(U, Q, edges1, edges2, X0=None, **kwargs):
    """ Alternating direction graph matching for directed graphs. Undirected graphs
    can be converted to directed ones by duplicating the edges. Compared to `ADGM`,
    here the structures of the graphs are given by `edges1` and `edges2` instead
    of a single sparse matrix `P`.
    Args:
        - `U` (`numpy.ndarray`): An `n1 x n2` array representing the unary potentials, 
        i.e., the costs of matching individual nodes. For example, `U[i,p]` can be the 
        dissimilarity between the node `i` of `G1` and the node `p` of `G2`. The more 
        similar the two nodes are, the smaller the matching cost should be.
        - `Q` (`numpy.ndarray`): An `E1 x E2` array representing the pairwise
        (quadratic) potentials, where `E1 = len(edges1)` and `E2 = len(edges2)`.
        `Q[e1, e2]` represents the dissimilarity between the edge e1 of edges1
        and e2 of edges2.
        - `edges1` (`numpy.ndarray`): An `E1 x 2` array representing the edges
        of the first graph.
        - `edges2` (`numpy.ndarray`): An `E2 x 2` array representing the edges
        of the second graph.
        - `X0` (`numpy.ndarray`, optional): An `n1 x n2` array used for initializing ADGM.
        - `kwargs` (`dict`, optional): ADGM optimization parameters.
    Return:
        An `n1 x n2` binary assignment matrix
    """
    n1, n2 = U.shape
    N = n1*n2

    assert edges1.shape[1] == 2
    assert edges2.shape[1] == 2

    E1 = edges1.shape[0]
    E2 = edges2.shape[0]

    assert Q.shape == (E1, E2)

    # print(f'U {U.shape}:\n {U}')
    # print(f'Q {Q.shape}:\n {Q}')
    # print(f'edges1 {edges1.shape}:\n {edges1}')
    # print(f'edges2 {edges2.shape}:\n {edges2}')

    # Q = np.zeros((E1, E2))
    # Build the sparse symmetric pairwise potential matrix
    data, row, col = build_coo_potentials(Q, edges1, edges2, n2)
    
    P = coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    scheme = kwargs.get('scheme', 'adaptive')
    projection = kwargs.get('projection', 'euclidean')
    if scheme == 'adaptive':
        return ADGM_DenseMatches(U, P, X0=X0, **kwargs)
    elif scheme == 'fixed':
        rho = kwargs['rho']
        steps = kwargs['max_iter']
        if projection == 'euclidean':
            return ADGM_DenseMatches_Fixed(U, P.data, P.indices, P.indptr, P.shape, rho=rho, steps=steps)
        elif projection == 'bregman':
            return ADGM_DenseMatches_Fixed_Bregman(U, P.data, P.indices, P.indptr, P.shape, rho=rho, steps=steps)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
