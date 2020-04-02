'''Cell-cell variation measurements'''

import numpy as np
import pandas as pd
import scanpy.api as sc
import anndata
from typing import Union, Callable, Iterable
import matplotlib.pyplot as plt


def median_filter(x: np.ndarray,
                  k: int,
                  pad_ends: bool = True,) -> np.ndarray:
    '''Computes a median filter on signal `x` with
    specified kernel and stride parameters

    Parameters
    ----------
    x : np.ndarray
        [T,] length signal.
    k : int
        size of the kernel window. must be odd.

    Returns
    -------
    y : np.ndarray
        [T,] median filtered output.
        where the ends of the valid signal are padded by
        repeating initial and final values.

    Notes
    -----
    size of an output from a convolution with no zero-padding 
    and variable strides is:

    .. math::

        O = (I - k)/s + 1

    where O is the output size, I is the input size, k is the kernel,
    and s is the stride.

    So below:

    .. math::

        O = (T - k)/1 + 1

    References
    ----------
    https://arxiv.org/abs/1603.07285
    '''
    if k % 2 != 1:
        raise ValueError('k must be odd, you passed %d' % k)

    T = x.shape[0]
    O = np.zeros((T-k) + 1)

    sidx = k//2
    eidx = T - k//2

    for i, idx in enumerate(range(sidx, eidx)):
        m = np.median(x[idx:(idx+k)])
        O[i] = m

    if pad_ends:
        # hold values of filtered signal in H
        # and remake O as the padded output
        H = O.copy()
        O = np.zeros(x.shape[0])
        O[sidx:eidx] = H
        O[:sidx] = H[0]
        O[eidx:] = H[-1]
    return O


def diff_from_median(X: np.ndarray,
                     gene_names: np.ndarray,
                     min_mean: float = 0.1,
                     max_mean: float = 5.,
                     kernel_size: int = 49,
                     plot: bool = True,
                     logged: bool = True,) -> pd.DataFrame:
    '''Implements the difference-from-the-median
    method of estimating overdispersion

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] expression matrix.
    gene_names : np.ndarary
        [Genes,] gene name strings.
    min_mean : float
        minimum mean expression value to be included
        in the median variation calculation.
    max_mean : float
        maximum mean expression value to be included in the 
        median variation calculation.
    kernel_size : int
        size of the kernel for median filtering
        coefficients of variation.
    plot : bool
        plot the rolling median
    logged : bool
        values in `X` are log counts.

    Returns
    -------
    overdispersion : pd.DataFrame
        [Genes,] overdispersion estimate. indexed by gene name.

    References
    ----------
    Kolodziejczyk, A. A., et. al. (2015). Cell stem cell, 17(4), 471-85.
    '''
    gene_means = X.mean(axis=0)
    expressed_bidx = np.logical_and(
        gene_means > min_mean,
        gene_means < max_mean,)
    expr_gene_means = gene_means[expressed_bidx]

    expr_gene_names = gene_names[expressed_bidx]

    sdevs = X.std(axis=0)
    expr_sdevs = sdevs[expressed_bidx]
    cvs = expr_sdevs / expr_gene_means
    cvs2 = cvs**2

    if not logged:
        log_expr_means = np.log10(expr_gene_means)
        log_cv2 = np.log10(cvs2)
    else:
        log_expr_means = expr_gene_means
        log_cv2 = cvs2

    log_cv2_sorted = log_cv2[np.argsort(log_expr_means)]
    sorted_gene_names = expr_gene_names[np.argsort(log_expr_means)]
    # use a median filter to calculate the rolling median
    rolling_median = median_filter(log_cv2_sorted, k=kernel_size)

    # compute "difference from the median"
    DM = log_cv2_sorted - rolling_median
    ordered_DM = np.zeros_like(DM)
    ordered_DM[np.argsort(log_expr_means)] = DM
    ordered_gene_names = np.zeros_like(sorted_gene_names)
    ordered_gene_names[np.argsort(log_expr_means)] = sorted_gene_names

    df = pd.DataFrame({'DM': ordered_DM,
                       'Mean': expr_gene_means},
                      index=ordered_gene_names)

    if plot:
        low_dm_bidx = DM < np.percentile(ordered_DM, 95)
        plt.figure()
        plt.scatter(np.sort(log_expr_means)[low_dm_bidx],
                    log_cv2_sorted[low_dm_bidx],
                    alpha=0.5, s=1.,
                    c='gray')
        plt.scatter(np.sort(log_expr_means)[~low_dm_bidx],
                    log_cv2_sorted[~low_dm_bidx],
                    alpha=0.5, s=1.,
                    c='black')
        plt.plot(np.sort(log_expr_means), rolling_median,
                 c='blue', label='Median')
        plt.legend(frameon=False)
        plt.xlabel(r'$\mu$')
        plt.ylabel(r'CV^2')

    return df


def diff_from_centroid(adata: anndata.AnnData,
                       groupby: str = 'cell_type',
                       embedding: str = 'counts') -> pd.DataFrame:
    '''
    Computes the distance of each sample from the centroid
    of its group.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes]
    groupby : str
        column in `adata.obs` defining groups.
    embedding : str
        space in which to compute distances
        ["counts", "pca"].

    Returns
    -------
    df : pd.DataFrame
        [Cells, (DistanceToMean, Group)]
    '''

    if embedding == 'counts':
        X = adata.X.toarray()
    elif embedding == 'pca':
        X = adata.obsm['X_pca']
    else:
        raise ValueError()
    groups = np.unique(adata.obs[groupby])

    distances = []
    for i, g in enumerate(groups):
        group_bidx = np.array(adata.obs[groupby] == g)
        group_X = X[group_bidx, :]

        group_center = group_X.mean(0)  # [Genes,]
        group_center = group_center.reshape(1, -1)  # [1, Genes]
        group_center_mat = np.tile(
            group_center, (group_X.shape[0], 1))  # [Cells, Genes]

        D = np.sqrt(np.sum((group_X - group_center_mat)**2, axis=1))
        dist_df = pd.DataFrame(
            {'DistanceToMean': D,
             'Group': g,
             },
            index=np.array(adata.obs_names)[group_bidx],
        )
        distances += [dist_df]
    distances = pd.concat(distances, 0)
    return distances
