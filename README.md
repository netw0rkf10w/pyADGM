# Alternating Direction Graph Matching (ADGM)

This is a Python implementation of Alternating Direction Graph Matching (ADGM),
which was introduced in the paper
[Alternating Direction Graph Matching](https://khue.fr/publication/adgm/) (CVPR 2017)
by [D. Khuê Lê-Huu](https://khue.fr)
and [Nikos Paragios](http://cvn.centralesupelec.fr/~nikos).

A C++ implementation (with MATLAB wrapper) for hyper-graphs can be found here: https://github.com/netw0rkf10w/ADGM. I will add support for hyper-graphs to this repo in the future.
 
If you use any part of this code, please cite:
```
@inproceedings{lehuu2017adgm,
 title={Alternating Direction Graph Matching},
 author={L{\^e}-Huu, D. Khu{\^e} and Paragios, Nikos},
 booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})},
 year = {2017}
}
```


# Dependencies
In addition to some standard packages, it is required to intall [Numba](https://numba.pydata.org). It is strongly recommended to use a virtual environment such as `conda`, `virtualenv`, or `venv`. 


```
conda install numba
```

To install Numba in other environments:

```
pip install numba
```


# Usage

Consider two graphs `G1 = (V1, E1)` and `G2 = (V2, E2)`. Denote by `n1 = |V1|` 
and `n2 = |V2|` the corresponding numbers of nodes. For matching `G1` and `G2`, 
one needs to construct the corresponding unary and pairwise potentials 
(see [Examples](#examples)) and call ADGM as follows:

```python
X = ADGM(U, P, assignment_mask=None, X0=None, **kwargs)
```

The output `X` is an `n1 x n2` discete assignment matrix. 
The input arguments are described below.


- `U` (`numpy.ndarray`): An `n1 x n2` array representing the unary potentials, 
i.e., the costs of matching individual nodes. For example, `U[i,p]` can be the 
dissimilarity between the node `i` of `G1` and the node `p` of `G2`. The more 
similar the two nodes are, the smaller the matching cost should be.
- `P` (`numpy.ndarray` or `scipy.sparse.csr_matrix`): An `A x A` array 
representing the pairwiste potentials, where `A` is the number of non-zeros 
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
```python
    kwargs = {'rho': max(10**(-60.0/np.sqrt(n1*n2)), 1e-4),
              'rho_max': 100,
              'step': 1.2,
              'precision': 1e-5,
              'decrease_delta': 1e-3,
              'iter1': 5,
              'iter2': 10,
              'max_iter': 10000,
              'verbose': False}
```

# Examples

We give some examples of using ADGM for matching two sets of synthetic feature points. To reproduce the following results, run `demo.py`.

## Input feature points 

The following code generate two set of 2D points.

```python
import numpy as np
import random

# for reproducibility
np.random.seed(13)
random.seed(13)

# numbers of points
n1 = 40
n2 = 30
# in this example, we randomly take some points of the first set
# then randomly transform them, thus we need n1 >= n2
assert n1 >= n2

# create a set of randoms 2D points, zero-centered them
points1 = np.random.randint(100, size=(n1, 2))
points1 = points1 - np.mean(points1, axis=0)

# randomly transform it using a similarity transformation

# first construct the transformation matrix
theta = np.random.rand()*np.pi/2
# scale = np.random.uniform(low=0.5, high=1.5)
scale = 0.9
tx = np.random.randint(low=120, high=150)
ty = np.random.randint(50)
M = np.array([[scale*np.cos(theta), np.sin(theta),       tx],
                [-np.sin(theta),      scale*np.cos(theta), ty]])

# then transform the first set of points
points2 = np.ones((3, n1))
points2[:2] = np.transpose(points1)
points2 = np.transpose(np.dot(M, points2))

# randomly keep only n2 points
indices = list(range(n1))
random.shuffle(indices)
points2 = points2[indices]
points2 = points2[:n2]

# we can add random potential match candidates
# assignment_mask = np.logical_or(np.random.randn(n1, n2) > 0.5, X_gt)
# but for now let's use all the candidates
assignment_mask = None

# ground-truth matching, for evaluation
X_gt = np.zeros((n1, n2), dtype=int)
for idx2, idx1 in enumerate(indices[:n2]):
    X_gt[idx1, idx2] = 1
```

Let us visualize the features and the ground-truth matching.

```python
import os
from utils import draw_results

# where to save outputs
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Visualize the feature points and the ground-truth matching
plot = plt.subplots()
plot[1].title.set_text('Ground-truth matching')
draw_results(plot, points1, points2, X=X_gt)
plt.show()
```

<img src="figures/ground-truth.jpg?raw=true" width="500px" alt="Ground-truth matching">


## Matching using fully-connected graphs

In this section, we try ADGM for matching the above set of points by defining
two **fully-connected** graphs. To this end, we will need to define the unary and pairwise potentials.

The unary potentials represent the cost of matching individual points (e.g., a dissimilarity measure). Since in this example, the points are just plain 2D
points with no attributes, we omit the unary potentials:

```python
# We do not use any unary potentials
U = np.zeros((n1, n2))
```

For any pair of match candidates `a = (i, p)` and `b = (j, q)` (see [Usage](#usage)
for notation), the pairwise potential `P[a, b]` can be defined, for example,
as a weighted sum of `d_length(ij, pq)` and `d_angle(ij, pq)`, which are
respectively the differences in length and in angle between the two vectors 
`ij` and `pq`: `P[a, b] = w * d_length(ij, pq) + (1 - w) * d_angle(ij, pq)`. Let `d_ij, d_pq` be the lengths of the vectors and `alpha` be the angle between them. One can, for example, define the above dissimilarity as `d_length(ij, pq) = |l_ij - l_pq|/(l_ij + l_pq) - 1` and `d_angle(ij, pq) = 0.5*(-cos(angle) - 1)`. Note that this is slightly different from what is proposed in the paper (Equations 39 and 40).

The above pairwise potentials are supported by the function `build_pairwise_potentials`. Once can thus build the graph matching problem 
and solve it using ADGM as follows:

```python
from adgm.adgm import ADGM
from adgm.energy import build_pairwise_potentials

# Weight of the length term (the weight of the angle term is thus 1-len_weight)
len_weight = 0.7

# Build the pairwise potentials with fully-connected graphs
P_dense = build_pairwise_potentials(points1, points2, 
                assignment_mask=assignment_mask, len_weight=len_weight)

# Call ADGM solver
kwargs = {'rho': max(10**(-60.0/np.sqrt(n1*n2)), 1e-4),
          'rho_max': 100,
          'step': 1.2,
          'precision': 1e-5,
          'decrease_delta': 1e-3,
          'iter1': 5,
          'iter2': 10,
          'max_iter': 10000,
          'verbose': False}
X_dense = ADGM(U, P_dense, **kwargs)

# Plot the results
plot = plt.subplots()
plot[1].title.set_text('Fully-connected graph matching')
draw_results(plot, points1, points2, X=X_dense, X_gt=X_gt)
plt.show()
```

<img src="figures/dense.jpg?raw=true" width="500px" alt="Dense matching">

<span style="color:green">Green</span>: Good matches (True positives). <span style="color:red">Red</span>: Bad matches (False positives). <span style="color:yellow">Yellow</span>: Missed matches (False negatives).
### Matching using sparse graphs

Instead of using fully-connected graphs, which is computationally expensive,
one can also use sparse graphs. A solution is to use approximate nearest neighbors to define graph edges. Below I give an example of using Delaunay triangulation.

The `build_pairwise_potentials` function has two arguments `adj1` and `adj2` for
representing the adjacency matrices of the two point sets.


```python
from scipy.spatial import Delaunay
from utils import get_adjacency_matrix

# Building the graphs based on Delaunay triangulation
tri1 = Delaunay(points1)
adj1 = get_adjacency_matrix(tri1)
tri2 = Delaunay(points2)
adj2 = get_adjacency_matrix(tri2)

# Build the pairwise potentials with Delaunay graphs
P_sparse = build_pairwise_potentials(points1, points2, adj1=adj1, adj2=adj2,
                assignment_mask=assignment_mask, len_weight=len_weight)

# Call ADGM solver
X_sparse = ADGM(U, P_sparse, assignment_mask=assignment_mask, **kwargs)
print('Sparse matching time (s):', time.time() - start)

# Plot the results
plot = plt.subplots()
plot[1].title.set_text('Sparse graph matching')
plt.triplot(points1[:,0], points1[:,1], tri1.simplices, color='b')
plt.triplot(points2[:,0], points2[:,1], tri2.simplices, color='b')
draw_results(plot, points1, points2, X=X_dense, X_gt=X_gt)
plt.show()
```

<img src="figures/sparse.jpg?raw=true" width="500px" alt="Sparse matching">


# Notes
1. The very first execution of the code will be a bit slow because Numba needs to
compile some functions at the first run (see [here](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html) for more details).

2. If the displacement of the sets of points are too large in terms of both scaling and rotation, it would be better to use sparse graphs instead of fully-connected ones.
