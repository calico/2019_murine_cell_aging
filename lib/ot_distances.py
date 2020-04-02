'''Optimal transport distances

Notes
-----
Currently uses munkres for an exact min-cost solution to linear 
sum assignment problem, where we set distance in some measurement
space as the cost function.
There is almost certainly some reasonable approximation to this
expensive computation, which scales `O(n^3)`.
'''
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist


def group_distance(X: np.ndarray,
                   idx0: np.ndarray,
                   idx1: np.ndarray,
                   metric: str = 'munkres_cost',
                   n_jobs: int = 1) -> float:
    '''Calculate the distance between two groups of points in `X`

    Parameters
    ----------
    X : np.ndarray
        [Cells, Features] embedding.
    idx0 : np.ndarray
        [Cells,] index of cells in group 0.
    idx1 : np.ndarray
        [Cells,] index of cells in group 1.
    metric : str
        distance metric to use.
        'centroid_distance_euclidean' - euclidean dist between centroids.
        'centroid_distance_manhattan' - manhattan dist between centroids.
        'centroid_distance_cosine' - cos dist between centroids.
        'average' - mean pairwise euclidean dist between members of 
            group `0` and members of group `1`.
        'median' - median pairwise euclidean dist between members of 
            group `0` and members of group `1`.
        'cosine' - mean pairswise cosine dist between members of 
            group `0` and members of group `1`.
        'munkres_cost' - munkres min cost solution for optimal transport
            of group `0` to group `1`.

    Returns
    -------
    dist : float
        distance between groups 0 and 1.
    '''
    # dm[i, j] is distance between X[i,:] and X[j,:]

    if metric.lower() == 'centroid_distance_euclidean':

        cent_0 = X[idx0, :].mean(0)
        cent_1 = X[idx1, :].mean(0)
        dist = np.sqrt(np.sum((cent_0 - cent_1)**2))

    elif metric.lower() == 'centroid_distance_manhattan':

        cent_0 = X[idx0, :].mean(0)
        cent_1 = X[idx1, :].mean(0)
        dist = np.sum(np.abs(cent_0 - cent_1))

    elif metric.lower() == 'centroid_distance_cosine':

        cent_0 = X[idx0, :].mean(0)
        cent_1 = X[idx1, :].mean(0)
        dist = pdist(cent_0, cent_1, metric='cosine')

    elif metric.lower() == 'average':
        x_0 = X[idx0, :]
        x_1 = X[idx1, :]
        dm = pairwise_distances(x_0, x_1, n_jobs=n_jobs, metric='euclidean')
        dist = np.mean(dm)

    elif metric.lower() == 'median':
        x_0 = X[idx0, :]
        x_1 = X[idx1, :]
        dm = pairwise_distances(x_0, x_1, n_jobs=n_jobs, metric='euclidean')
        dist = np.median(dm)

    elif metric.lower() == 'cosine':

        x_0 = X[idx0, :]
        x_1 = X[idx1, :]
        dm = pairwise_distances(x_0, x_1, n_jobs=n_jobs, metric='cosine')
        assert dm.shape[0] == len(x_0)
        assert dm.shape[1] == len(x_1)
        dist = np.mean(dm)

    elif metric.lower() == 'munkres_cost':

        x_0 = X[idx0, :]
        x_1 = X[idx1, :]
        dm = pairwise_distances(x_0, x_1, n_jobs=n_jobs, metric='euclidean')
        row_idx, col_idx = linear_sum_assignment(dm)
        dist = np.sum(dm[row_idx, col_idx])

    else:
        raise ValueError('arg to `metric` %s not understood' % metric)

    return dist


def random_sample(groupbys: list,
                  groups: list,
                  n: int = 500) -> np.ndarray:
    '''Generate a random index of cells where cells 
    are drawn from `groups` in `groupbys`. Multiple `groupbys`
    can be provided simultaneously for expedient random
    sampling of subpopulations.

    Parameters
    ----------
    groupbys : list
        list of [Cells,] vectors of group assignments for 
        each group to sample from.
        e.g. [
            np.array(['typeA', 'typeA', 'typeB']),
            np.array(['0', '1', '0']),
        ]
        where each of the arrays is a unique grouping variable.
    groups : list
        list of str values in groupbys to sample from.
        e.g. ['typeA', '0'] would sample only the first
        cell from the above `groupbys` example.
    n : int
        sample size.

    Returns
    -------
    idx : np.ndarray
        [n,] index of integers.

    Notes
    -----
    If sample size `n` is larger than the number of cells in a group,
    we reduce the sample size to `0.8*n_cells_in_group`.

    Examples
    --------
    >>> groupbys = [
    ...   adata.obs['cell_type'],
    ...   adata.obs['louvain'],
    ... ]
    >>> # sample only B cells in cluster '3'
    >>> groups = ['B cell', '3']
    >>> # returns indices of random B cells
    >>> # in louvain cluster '3'
    >>> idx = random_sample(groupbys, groups)
    '''
    assert len(groupbys) == len(groups)
    bidxs = []
    for i in range(len(groups)):
        group_bidx = (groupbys[i] == groups[i])
        bidxs.append(group_bidx)
    bidx = np.logical_and.reduce(bidxs)
    group_idx = np.where(bidx)[0].astype(np.int)

    if int(0.8*len(group_idx)) < n:
        print(f'Cannot draw {n} cells from group of size {len(group_idx)}.')
        n = 0.8*len(group_idx)
        print(f'Reducing sample size to {n}.')

    idx = np.random.choice(group_idx,
                           size=n,
                           replace=False).astype(np.int)
    return idx
