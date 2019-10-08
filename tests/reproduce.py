import numpy as np
import random
import numba

np.random.seed(12)
random.seed(12)

@numba.jit(nopython=True, parallel=True)
def build_pairwise_sparse(points1, points2, adj1=None, adj2=None, assignment_mask=None, len_weight=0.5):
    """
    points1: n1 x 2 array
    points2: n2 x 2 array
    For each pair of matches: i <--> p and j <--> q, the pairwise cost
        P(ip, jq) = w*leng + (1 - w)*0.5*(1 - cos_angle)
    where 
        leng = np.abs(norm1 - norm2)/(norm1 + norm2)
        cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
        norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
        norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
        w = len_weight
    adj1, adj2: n1xn1, n2xn2 adjacency matrices of points1 and points2.
        If None then fully connected.
    assignment_mask: the mask for the assignment matrix
        assignment_mask[i, p] = 0 means i should never match to p
    """
    len_weight = min(1.0, max(len_weight, 0))
    n1 = len(points1)
    n2 = len(points2)
    N = n1*n2
    
    # all assignment indices
    # if assignment_mask is None:
    #     assignment_mask = np.ones((n1, n2))
    aidx = np.flatnonzero(assignment_mask)
    A = len(aidx)

    # pre-compute x,y coordinates of each linear index 
    lidx = np.zeros(A)
    ridx = np.zeros(A)
    for idx_a in range(A):
        a = aidx[idx_a]
        p = a % n2
        i = int((a - p)/n2)
        lidx[idx_a] = i
        ridx[idx_a] = p

    # count the number of non-zero element in the final pairwise matrix
    # not easy to get a formula, so we iterate
    # upper-bound for the total number of non-zeros: min(2*E1*E2, A*(A-1)/2)
    num_nonzeros = 0
    for idx_a in range(A - 1):
        i = lidx[idx_a]
        p = ridx[idx_a]
        for idx_b in range(idx_a + 1, A):
            j = lidx[idx_b]
            q = ridx[idx_b]
            if q == p or j == i:
                continue
            # if ij or pq is not an edge then no potential
            if adj1 is not None and adj1[i, j] <= 0:
                continue
            if adj2 is not None and adj2[p, q] <= 0:
                continue
            num_nonzeros += 1
    
    # now we iterate again to get the indices i, j, p, q
    # last column of data is the potential
    data = np.zeros((num_nonzeros, 5))
    idx = 0
    for idx_a in range(A - 1):
        i = lidx[idx_a]
        p = ridx[idx_a]
        for idx_b in range(idx_a + 1, A):
            j = lidx[idx_b]
            q = ridx[idx_b]
            if q == p or j == i:
                continue
            # if ij or pq is not an edge then no potential
            if adj1 is not None and adj1[i, j] <= 0:
                continue
            if adj2 is not None and adj2[p, q] <= 0:
                continue
            data[idx, 0] = i
            data[idx, 1] = j
            data[idx, 2] = p
            data[idx, 3] = q
            idx += 1


    for pidx in numba.prange(num_nonzeros):
        i = int(data[pidx, 0])
        j = int(data[pidx, 1])
        p = int(data[pidx, 2])
        q = int(data[pidx, 3])
        xi = points1[int(i), 0]
        yi = points1[int(i), 1]
        xj = points1[int(j), 0]
        yj = points1[int(j), 1]
        xp = points2[int(p), 0]
        yp = points2[int(p), 1]
        xq = points2[int(q), 0]
        yq = points2[int(q), 1]

        norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
        norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
        leng = np.abs(norm1 - norm2)/(norm1 + norm2)
        cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
        tt = len_weight*leng + (1 - len_weight)*0.5*(1 - cos_angle)

        data[pidx, 4] = tt

    return data


def main():
    n1 = 4
    n2 = 2
    assert n1 >= n2

    # create a set of randoms 2D points, zero-centered them
    points1 = np.random.randint(100, size=(n1, 2))
    points1 = points1 - np.mean(points1, axis=0)

    # randomly transform it
    theta = np.random.rand()*np.pi/2
    # scale = np.random.uniform(low=0.5, high=1.5)
    scale = 1
    tx = np.random.randint(low=120, high=150)
    ty = np.random.randint(50)
    M = np.array([[scale*np.cos(theta), np.sin(theta),       tx],
                  [-np.sin(theta),       scale*np.cos(theta),  ty]])
    # transform the first set of features
    points2 = np.ones((3, n1))
    points2[:2] = np.transpose(points1)
    points2 = np.transpose(np.dot(M, points2))
    
    # randomly choose n2 points
    indices = list(range(n1))
    random.shuffle(indices)
    points2 = points2[indices]
    points2 = points2[:n2]
    
    # start = time.time()
    # adj1 = np.fill_diagonal(np.ones((n1, n1), dtype=int), 0)
    # adj2 = np.fill_diagonal(np.ones((n2, n2), dtype=int), 0)
    # assignment_mask = np.ones((n1, n2), dtype=int)
    # adj1 = np.fill_diagonal(np.ones((n1, n1)), 0)
    # adj2 = np.fill_diagonal(np.ones((n2, n2)), 0)
    assignment_mask = np.ones((n1, n2))
    # data = build_pairwise_sparse(points1, points2,
    #                              adj1=adj1,
    #                              adj2=adj2,
    #                              assignment_mask=assignment_mask
    #                              )

    data = build_pairwise_sparse(points1, points2,
                                 adj1=None,
                                 adj2=None,
                                 assignment_mask=assignment_mask
                                 )

    print(data)

@numba.jit(nopython=True)
def test():
    v = np.zeros(3, dtype=int)

if __name__ == "__main__":
    main()
    # test()