import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def get_adjacency_matrix(tri):
    """Adapted from https://github.com/danielegrattarola/spektral
    Input: scipy.spatial.Delaunay triangulation
    Output: adjacency_matrix
    """
    # Important: edges may contain duplicates
    edges = np.concatenate((tri.vertices[:, :2],
                            tri.vertices[:, 1:],
                            tri.vertices[:, ::2]), axis=0)
    n = tri.points.shape[0]
    adj = np.zeros((n, n), dtype=np.int64)
    adj[edges[:, 0], edges[:, 1]] = 1
    return np.clip(adj + adj.T, 0, 1)

def get_matches(X):
    """
    return a list of matches (i, p) from the assignment matrix X
    """
    return np.transpose(np.nonzero(X))

def draw_matches(plot, points1, points2, matches=None, colorm='g', s=50, linewidth=2):
    fig, ax = plot
    # Draw point based on above x, y axis values.
    plt.scatter(points1[:, 0], points1[:, 1], s=s)
    plt.scatter(points2[:, 0], points2[:, 1], s=s)
    if matches is not None:
        lines = []
        for i1, i2 in matches:
            lines.append([points1[i1], points2[i2]])

        colors = [colorm]*len(lines)
        lc = mc.LineCollection(lines, colors=colors, linewidths=linewidth)
        ax.add_collection(lc)


def draw_results(plot, points1, points2, X=None, X_gt=None):
    """
    Draw good matches (true positives) in green, bad matches (false positives)
    in red, and missed matches (false negatives) in yellow. If X_gt is not given
    then draw all matches in green. If in addition, X is not given, then draw
    only the points
    Args:
        X: assignment matrix
        X_gt: ground-truth assignment matrix
    """
    if X is None:
        draw_matches(plot, points1, points2, matches=None)
    elif X_gt is None:
        draw_matches(plot, points1, points2, matches=get_matches(X))
    else:
        # Draw true positives in green
        draw_matches(plot, points1, points2, matches=get_matches(X & X_gt), colorm='g')
        # Draw false positives in red
        draw_matches(plot, points1, points2, matches=get_matches(X & ~X_gt), colorm='r')
        # Draw false negatives in yellow
        draw_matches(plot, points1, points2, matches=get_matches(~X & X_gt), colorm='y')
