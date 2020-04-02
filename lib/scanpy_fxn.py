'''
Assorted utilities for scanpy object interaction
'''
import numpy as np
import pandas as pd
import scanpy.api as sc
import os
from typing import Union, Callable, Iterable
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import anndata

'''Preprocessing'''


def remove_poor_quality_cells(exp: anndata.AnnData,
                              min_genes: int = 300,
                              max_genes: int = 2500,
                              max_mito: float = 0.1,
                              max_rrna: float = 0.05,
                              min_cells: int = 10,
                              mito_prefix: str = 'mt-',
                              plot: bool = False,
                              save_plot: str = None,
                              ) -> anndata.AnnData:
    '''
    Parameters
    ----------
    exp : anndata.AnnData
    min_genes : int
        minimum genes per cell to keep
    min_counts : int
        min counts per cell to keep
    max_mito : float
        [0, 1] maximum proportion of reads in a cell allowed to map
        to the mitochondrial genome before the cell is filtered out.
    max_rrna : float
        [0, 1] maximum proportion of reads in a cell allowed to map
        to the Rn45s locus before the cell is filtered out. 
    plot : bool
        perform plotting.
    save_plot : str
        path for plot exports.
    '''
    if save_plot is not None:
        if not os.path.exists(save_plot):
            os.mkdir(save_plot)

    # use the Tabula Muris QC cutoffs
    sc.pp.filter_cells(exp, min_genes=500)
    sc.pp.filter_cells(exp, min_counts=1000)
    if min_cells is not None:
        sc.pp.filter_genes(exp, min_cells=min_cells)
    else:
        # don't filter genes, as we want to maintain a common set for
        # downstream cell type classification
        pass

    # remove cells with high mitochondrial gene fraction
    # see: Classification of low quality cells from single-cell RNA-seq data
    #      https://doi.org/10.1186/s13059-016-0888-1
    try:
        mito_genes = exp.var_names.str.startswith(mito_prefix)
    except AttributeError:
        print('Inferred type probably was not `string`. Parsing with list comprehension.')
        mito_bidx = np.array([True if x[:len(
            mito_prefix)] == mito_prefix else False for x in exp.var_names], dtype=np.bool)
        mito_genes = pd.Categorical(exp.var_names[mito_bidx])
        print('MITO GENES')
        print(mito_genes)
    # for each cell compute fraction of counts in mito genes vs. all genes
    # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
    exp.obs['percent_mito'] = np.sum(
        exp[:, mito_genes].X, axis=1).A1 / np.sum(exp.X, axis=1).A1
    exp.obs['percent_Rn45s'] = exp[:, 'Rn45s'].X / np.sum(exp.X, axis=1).A1
    # add the total counts per cell as observations-annotation to adata
    exp.obs['n_counts'] = exp.X.sum(axis=1).A1

    if plot:
        if save_plot is not None:
            vln_name = 'vln_qc_before.png'
        else:
            vln_name = None

        sc.pl.violin(exp, ['n_genes', 'n_counts', 'percent_mito', 'percent_Rn45s'],
                     jitter=0.3, multi_panel=True, save=vln_name)

    if max_genes is not None:
        exp = exp[exp.obs['n_genes'] < max_genes, :]
    exp = exp[exp.obs['percent_mito'] < max_mito, :]
    exp = exp[exp.obs['percent_Rn45s'] < max_rrna, :]

    if plot:
        if save_plot is not None:
            vln_name = 'vln_qc_after.png'
        else:
            vln_name = None

        sc.pl.violin(exp, ['n_genes', 'n_counts', 'percent_mito', 'percent_Rn45s'],
                     jitter=0.3, multi_panel=True, save=vln_name)
    return exp


def remove_rrna(adata: anndata.AnnData,
                rrna_names: list = ['Gm42418', 'AY036118', 'Rn45s']) -> anndata.AnnData:
    '''Remove transcripts which overlap with rRNA annotations
    and likely contain rRNA aligning reads.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] experiment.
    rrna_names : list
        str names of variables in `adata.var_names` to be removed.
    '''
    rm_bidx = np.array(
        [True if x in rrna_names else False for x in adata.var_names])
    keep_bidx = np.logical_not(rm_bidx)
    adata = adata[:, keep_bidx]
    return adata


def preprocess_and_export(exp: anndata.AnnData,
                          out_path: str,
                          plot_path: str = None,
                          **kwargs) -> None:
    '''
    Preprocess an experiment file by removing low quality cells,
    performing normalization, and simple dimensionality reduction.

    Parameters
    ----------
    exp : anndata.AnnData
        AnnData object of a single cell experiment.
    out_path : str
        path to an output Loom file.
    plot_path : str
        path for plots.

    Returns
    -------
    None.
    '''
    assert os.path.exists(os.path.split(out_path)[0])
    exp.var_names_make_unique()
    exp = remove_poor_quality_cells(exp, plot=True, save_plot=plot_path,
                                    **kwargs)
    exp = remove_rrna(exp,)
    print('Saving object to %s' % out_path)
    exp.write_loom(out_path)
    return


'''Clustering Utilities'''


def silhouette_index(adata: anndata.AnnData,
                     cluster_partition: Union[str, list, tuple],
                     embedding: str = 'pca') -> np.ndarray:
    '''Compute silhoutte indices for a cluster partition

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object. Must contain `cluster_partition` and 
        `groupby` in `.obs_keys`.
    cluster_partition : str, tuple, list
        key or keys in `.obs_keys` cluster partition assignments.
        if multiple keys are passed, provides silhouette values 
        for each.

    Returns
    -------
    silhouettes : np.ndarray
        [N,] index values.

    References
    ----------
    Silhouettes: A graphical aid to the interpretation 
    and validation of cluster analysis
    Peter J.Rousseeuw
    Journal of Computational and Applied Mathematics, 1987
    '''
    from sklearn.metrics import silhouette_score
    if embedding == 'pca':
        X = adata.obsm['X_pca']
    elif embedding == 'counts':
        X = adata.X.toarray()
    else:
        raise ValueError('%s is an invalid embedding.' % embedding)

    if type(cluster_partition) == str:
        cluster_partition = [cluster_partition]

    silhouettes = np.zeros(len(cluster_partition))
    for i, cp in enumerate(cluster_partition):
        s = silhouette_score(
            X,
            labels=adata.obs[cp],
            metric='euclidean',
        )
        silhouettes[i] = s
    return silhouettes


def cluster_distribution(adata: anndata.AnnData,
                         cluster_partition: str,
                         groupby: str,) -> pd.DataFrame:
    '''Quantifies the proportion of a group in each subset of
    a given cluster partition.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object. Must contain `cluster_partition` and 
        `groupby` in `.obs_keys`.
    cluster_partition : str
        key in `.obs_keys` cluster partition assignments.
    groupby : str
        key in `.obs_keys` specifying a group of interest.
        The relative proportion of each element in this group is computed
        for each element in `cluster_partition`.

    Returns
    -------
    distributions : pd.DataFrame
        [Groups, Clusters] each i, j := number of group i in cluster j

    Sets `adata.uns[cluster_distribution][cluster_partition][groupby]`.
    '''
    cluster_idx = adata.obs[cluster_partition]
    group_idx = adata.obs[groupby]

    clusters = sorted(list(set(cluster_idx)))
    groups = sorted(list(set(group_idx)))

    # [Groups, Clusters]
    # values are the number of cells in each group in a given cluster
    distributions = np.zeros((len(groups), len(clusters)))

    for i, g in enumerate(groups):
        g_bidx = group_idx == g
        g_cluster_idx = cluster_idx[g_bidx]

        for j, c in enumerate(clusters):
            distributions[i, j] = np.sum(g_cluster_idx == c)

    distributions = pd.DataFrame(
        distributions,
        index=groups,
        columns=clusters)
    adata.uns['cluster_distribution'] = defaultdict(dict)
    adata.uns['cluster_distribution'][cluster_partition] = defaultdict(dict)
    adata.uns['cluster_distribution'][cluster_partition][groupby] = distributions
    return distributions


def plot_cluster_distribution(
        distributions: pd.DataFrame,
        figsize: tuple = (6, 4),
        proportion: bool = False,
        **kwargs) -> matplotlib.axes.Axes:
    '''
    Plot a distribution of groups across clusters

    Parameters
    ----------
    distributions : pd.DataFrame
        [Groups, Clusters] each i, j := proportion of group i in cluster j.
        indexs are group names, column names are cluster names.
    figsize : tuple
        [H, W] of plot.
    proportion : bool
        plot proportions rather than counts.

    Returns
    -------
    matplotlib.axes.Axes
    '''
    if proportion:
        yvar = 'Proportion'
        idx = distributions.index
        cols = distributions.columns
        normmat = np.tile(
            np.sum(np.array(distributions), 1).reshape(-1, 1), (1, distributions.shape[1]))
        distributions = np.array(distributions) / normmat
        distributions = pd.DataFrame(distributions, index=idx, columns=cols)
    else:
        yvar = 'Count'

    dfm = pd.melt(distributions.reset_index(),
                  id_vars='index')
    dfm.columns = ['Group', 'Cluster', yvar]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(data=dfm, x='Cluster', y=yvar, hue='Group',
                ax=ax, **kwargs)
    sns.despine()
    ax.legend(frameon=False, bbox_to_anchor=(1., 1.))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    plt.tight_layout()
    return ax


def cluster_distribution_conttab(
        distributions: Union[np.ndarray, pd.DataFrame],
) -> (float, float):
    '''Test cluster distributions for significant differences
    using the Chi2 test for contingency tables

    Parameters
    ----------
    distributions : pd.DataFrame
        [Groups, Clusters] each i, j := proportion of group i in cluster j.
        indexs are group names, column names are cluster names.

    Returns
    -------
    chi2 : float
        value of the chi2 statistic.
    p : float
        p-value.
    '''
    tab = np.array(distributions)
    chi2, p, degfree, expected = stats.chi2_contingency(tab)
    return chi2, p


'''Cell Type Proportions'''


def get_cell_states_per_age(exp: anndata.AnnData,
                            state_var: str = 'cell_type',) -> pd.DataFrame:
    '''Count the number and proportion of each cell state captured for each
    replicate in each age

    Parameters
    ----------
    exp : anndata.AnnData
        `.obs` must contain [`state_var`, age, animal, replicate]
    state_var : str
        state variable name.

    Returns
    -------
    df : pd.DataFrame
        columns ["Cell State", "Number", "Proportion", "Age",]
        where each row is a single 10X library (one replicate in one animal)
    '''
    obs = exp.obs.copy()

    cell_states = obs.groupby(['age',
                               state_var,
                               'animal',
                               'replicate']).count().loc[('old'), 'batch'].index.levels[0]
    n_young = np.array(obs.groupby(['age',
                                    state_var,
                                    'animal',
                                    'replicate']).count().loc['young', 'batch'])
    n_old = np.array(obs.groupby(['age',
                                  state_var,
                                  'animal',
                                  'replicate']).count().loc['old', 'batch'])
    n_young_sum_replicate = np.concatenate(
        [np.array(
            obs.groupby(['age', 'animal', 'replicate']).count().loc['young', 'batch'])]*len(cell_states)
    )
    n_old_sum_replicate = np.concatenate(
        [np.array(
            obs.groupby(['age', 'animal', 'replicate']).count().loc['old', 'batch'])]*len(cell_states)
    )

    df = pd.DataFrame(
        {'Cell State': np.concatenate(
            2*[np.repeat(cell_states, len(np.unique(obs['animal']))*len(np.unique(obs['replicate'])))]),
         'Number': np.concatenate([n_young, n_old]),
         'Proportion': np.concatenate([n_young/n_young_sum_replicate, n_old/n_old_sum_replicate]),
         'Age': pd.Categorical(['Young']*len(n_young)+['Old']*len(n_old)),
         })
    df = df.loc[~np.isnan(df['Number']), :]
    return df
