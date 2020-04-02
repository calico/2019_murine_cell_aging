'''Differential Expression'''

import numpy as np
import pandas as pd
import scanpy.api as sc
import anndata
import tqdm
from typing import Union, Callable, Iterable

from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def rank_genes_groups2df(markers: dict,) -> pd.DataFrame:
    '''Convert `scanpy.api.tl.rank_genes_groups` output to 
    a more tractable `pd.DataFrame`

    Parameters
    ----------
    markers : dict
        output from `scanpy.api.tl.rank_genes_groups`, as stored
        in `anndata.uns[some_key_name]`.

    Returns
    -------
    df : pd.DataFrame
        ['gene', 'log2_fc', 'q_val']
    '''
    logfc = np.array([x[0] for x in markers['logfoldchanges']])
    names = np.array([x[0] for x in markers['names']])
    qvals = np.array([x[0] for x in markers['pvals_adj']])

    # use gene names as indices for convenience.
    df = pd.DataFrame({
        'gene': names,
        'log2_fc': logfc,
        'q_val': qvals,
    },
        index=names)

    return df


def logistic_regression_lr_test(_input: np.ndarray,
                                _labels: np.ndarray) -> (float, float):
    '''
    Performs a likehood ratio test on the significance of features in 
    `_input` for predicting binary classes in `_labels`.

    Parameters
    ----------
    _input : np.ndarray
        [N, P] where N is the number of samples P is the number of features.
    _labels : np.ndarray
        [N,] binary labels

    Returns
    -------
    likelihood_ratio : float
        likelihood ratio between a fit logistic regression model and a null
        model that simply predicts the expectation of class 1 for each sample.
    p : float
        p-value from the likelihood ratio test using the chi2 distribution with 
        df == P,

    Notes
    -----
    Useful as a differential expression measurement.

    References
    ----------
    Identification of transcriptional signatures for cell types from single-cell RNA-Seq.
    Vasilis Ntranos, Lynn Yi, Pall Melsted, Lior Pachter.
    https://www.nature.com/articles/s41592-018-0303-92
    '''
    if len(np.unique(_labels)) != 2:
        msg = 'labels must be binary.'
        raise ValueError(msg)

    model = LogisticRegression(solver='lbfgs')
    model.fit(_input, _labels)
    P = _input.shape[1]  # number of features

    _preds = np.array(model.predict_proba(_input))[
        :, 1]  # len(N), prob(class1)
    # log Likelihood = \sum y*log(model(x)) + (1 - y)*log(1-model(x))
    model_log_likelihood = np.sum(
        _labels*np.log(_preds) + (1-_labels)*np.log(1-_preds)
    )
    # null model votes the observed frequency of class 1 for all items
    prob_class_1 = np.sum(_labels) / _labels.shape[0]
    null_log_likelihood = np.sum(
        _labels*np.log(prob_class_1) + (1 - _labels)*np.log(1 - prob_class_1)
    )
    # log likelihood ratio
    likelihood_ratio = model_log_likelihood - null_log_likelihood
    p = 1 - stats.chi2.cdf(2*likelihood_ratio, P)  # location, df
    return likelihood_ratio, p


def diffex_multifactor(adata: anndata.AnnData,
                       grouping_0: str,
                       grouping_1: str,
                       use_raw: bool = False,
                       method: str = 'wilcoxon',
                       min_cells: int = 10,
                       **kwargs) -> pd.DataFrame:
    '''
    Perform differential expression analysis across a top
    level and sub level contrast factor in an AnnData experiment.

    Parameters
    ----------
    adata : anndata.AnnData
        single cell experiment data.
    grouping_0 : str
        key in `exp.obs` to use for the top level contrast.
    grouping_1 : str
        key in `exp.obs` to use for the lower level contrast.
    use_raw : bool
        use `adata.raw.X` for expression values rather than `adata.X`.
    method : str
        statistic test to use.
        'wilcoxon' - use wilcoxon rank sums test.
        'logreg' - use logistic regression and a likelihood ratio test.
    min_cells : int
        minimum number of cells within each subgroup of `grouping_0` that 
        must express a gene in order for diffex testing to be performed.

    Returns
    -------
    df : pd.DataFrame
        Columns:
         ['top_group', 
          'lower_group', 
          'gene', 
          'log2_fc',
          'ranksum_stat',
          'p_val', 
          'q_val', 
          'lower_group_mean', 
          'other_groups_mean']

    Notes
    -----
    Uses the Wilcoxon Rank-Sum test for significant differences
    and the Benjamini Hochberg procedure for FDR control.

    Compares one lower level group to all others within each 
    top level group.

    Postive signs of `log2_fc` indicate that `lower_group` expresses
    higher levels than `other_lower_groups`. Negative signs indicate
    the opposite relationship.

    Examples
    --------
    >>> dex = diffex_multifactor(adata, 'cell_type', 'age')
    '''
    if method not in ('wilcoxon', 'ranksums', 'logistic', 'logreg'):
        msg = f'{method} is not a valid differential expression method.'
        raise ValueError(msg)

    groups_0 = sorted(list(set(adata.obs[grouping_0])))
    groups_1 = sorted(list(set(adata.obs[grouping_1])))

    n_comparisons = len(groups_0)*len(groups_1)

    n_genes_being_tested = 0
    for i, top_group in tqdm.tqdm(enumerate(groups_0),
                                  desc='Predicting for each %s' % grouping_0):

        tg_exp = adata[adata.obs[grouping_0] == top_group, :].copy()
        gene_names_to_test = tg_exp.var_names.tolist()
        n_genes_being_tested += len(gene_names_to_test)
    print('%d genes will be tested in total.' % n_genes_being_tested)

    columns = [
        'top_group',
        'lower_group',
        'gene',
        'log2_fc',
        'test_statistic',
        'p_val',
        'q_val',
        'lower_group_mean',
        'other_groups_mean',
        'lower_group_n_expr',
        'other_groups_n_expr',
        'lower_group_n',
        'other_groups_n',
    ]

    df = pd.DataFrame(
        np.zeros((len(groups_1)*n_genes_being_tested, len(columns))),
        columns=columns,
    )

    k = 0
    for i, top_group in tqdm.tqdm(enumerate(groups_0),
                                  desc='Predicting for each %s' % grouping_0):

        tg_exp = adata[adata.obs[grouping_0] == top_group, :].copy()
        gene_names_to_test = tg_exp.var_names.tolist()

        for j, lower_group in enumerate(groups_1):

            lg_of_interest = tg_exp[tg_exp.obs[grouping_1]
                                    == lower_group, :].copy()
            all_other_lgs = tg_exp[tg_exp.obs[grouping_1]
                                   != lower_group, :].copy()

            if use_raw:
                lg_of_interest_X = lg_of_interest.raw.X
                all_other_lgs_X = all_other_lgs.raw.X
            else:
                lg_of_interest_X = lg_of_interest.X
                all_other_lgs_X = all_other_lgs.X

            if type(lg_of_interest_X) != np.ndarray:
                print('Converting to dense arrays...')
                lg_of_interest_X = lg_of_interest_X.toarray()
                all_other_lgs_X = all_other_lgs_X.toarray()
                print('Converted.')

            for g in range(len(gene_names_to_test)):
                gene = gene_names_to_test[g]
                x = lg_of_interest_X[:, g]
                y = all_other_lgs_X[:, g]

                # binary expression vectors
                x_bin = x > 0
                y_bin = y > 0

                # check explicitly if a gene meets the `min_cells` requirement,
                # record `None` values if it does not.
                if np.sum(x_bin) < min_cells and np.sum(y_bin) < min_cells:
                    df.iloc[k, :] = (top_group,
                                     lower_group,
                                     gene,
                                     None,
                                     None,
                                     None,
                                     None,
                                     x.mean(),
                                     y.mean(),
                                     int(x_bin.sum()),
                                     int(y_bin.sum()),
                                     len(x),
                                     len(y),
                                     )
                    k += 1
                    continue

                if method in ('wilcoxon', 'ranksums'):
                    W, p = stats.ranksums(x, y)
                elif method in ('logreg', 'logistic'):
                    _input = np.concatenate(
                        [x, y]).reshape(-1, 1)  # [Cells, 1]
                    _labels = np.array([0]*len(x) + [1]*len(y))
                    W, p = logistic_regression_lr_test(_input, _labels)
                else:
                    raise ValueError(
                        'method %s is not a valid option.' % method)
                if x.mean() == 0 or y.mean() == 0:
                    log2fc = None
                else:
                    log2fc = np.log2(x.mean() / y.mean())

                df.iloc[k, :] = (top_group,
                                 lower_group,
                                 gene,
                                 log2fc,
                                 W,
                                 p,
                                 p,
                                 x.mean(),
                                 y.mean(),
                                 int(x_bin.sum()),
                                 int(y_bin.sum()),
                                 len(x),
                                 len(y),
                                 )

                k += 1

            # perform multiple hypothesis test correction for this contrast

            dft = df.loc[~np.isnan(df['p_val']), :]
            ps = dft.loc[np.logical_and(dft['top_group'] == top_group,
                                        dft['lower_group'] == lower_group), 'p_val']

            _, qs, _, _ = multipletests(ps, alpha=0.05, method='fdr_bh')

            dft.loc[np.logical_and(dft['top_group'] == top_group,
                                   dft['lower_group'] == lower_group), 'q_val'] = qs
            df.loc[dft.index, 'q_val'] = dft.loc[:, 'q_val']

    return df


def volcano_plot(df: pd.DataFrame, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    '''Generate a volcano plot

    Parameters
    ----------
    df : pd.DataFrame
        differential expression output from `diffex_multifactor`.
    ax : matplotlib.axes.Axes

    Returns
    -------
    matplotlib.axes.Axes
    '''
    if 'significant' not in df.columns:
        print('Adding significance cutoff at alpha=0.05')
        df['significant'] = df['q_val'] < 0.05

    n_colors = len(np.unique(df['significant']))
    sns.scatterplot(data=df, x='log2_fc', y='nlogq', hue='significant',
                    linewidth=0.,
                    alpha=0.3,
                    ax=ax,
                    palette=sns.hls_palette(n_colors)[::-1])
    ax.set_xlim((-6, 6))
    ax.set_ylabel(r'$-\log_{10}$ q-value')
    ax.set_xlabel(r'$\log_2$ (Old / Young)')
    ax.get_legend().remove()
    return ax


def get_fold_change_matrix(adata: anndata.AnnData,
                           groupby: str,
                           contrast: str,
                           contrast_ref: int = 0) -> pd.DataFrame:
    '''Generate a [Cells, Genes] log2 fold-change matrix from
    an AnnData object

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes], contains `groupby` in the `.obs` pd.DataFrame.
    groupby : str
        column in `adata.obs` for categorical subsetting when calculating
        fold changes.
    contrast : str
        column for binary contrast, must be in `adata.obs`.
    contrast_ref : tuple
        index of group in `groupby` to use as the fold change reference.

    Returns
    -------
    fc_df : pd.DataFrame
        [Cells, Genes] matrix of log2 fold change values between
        levels in the `groupby`.
    '''
    assert contrast_ref in [0, 1], 'contrast_ref must be a binary int'
    assert len(adata.obs[contrast].unique()
               ) == 2, 'contrast must be a binary variable'
    if type(adata.X) != np.ndarray:
        X = adata.X.toarray()
    else:
        X = adata.X
    pl_df = pd.DataFrame(X, columns=adata.var_names)
    for k in [contrast, groupby]:
        pl_df.loc[:, k] = adata.obs[k].tolist()
    grped = pl_df.groupby([contrast, groupby]).mean()

    contrast_groups = adata.obs[contrast].unique()
    if contrast_ref == 0:
        cref_group = contrast_groups[0]
        ctest_group = contrast_groups[1]
    else:
        cref_group = contrast_groups[1]
        ctest_group = contrast_groups[0]

    # generate the fold change by groupby matrix
    fc_by_ct = np.array(grped.loc[ctest_group])/np.array(grped.loc[cref_group])
    # Test / Ref -> Inf when Ref is 0.
    # map this to the highest FC we see elsewhere in the plot
    fc_by_ct[np.isinf(fc_by_ct)] = np.max(fc_by_ct[~np.isinf(fc_by_ct)])
    # Test / Ref -> NaN when Test and Ref == 0.
    # map this value to 1., indicating no change in expression
    fc_by_ct[np.isnan(fc_by_ct)] = 1.
    # log2(0) is NaN. Set these values to the lowest value observed
    # elsewhere in the matrix to demonstrate a strong decrease in
    # expression
    fc_by_ct[fc_by_ct == 0.] = np.min(fc_by_ct[fc_by_ct > 0.])
    fc_by_ct = np.log2(fc_by_ct)
    fc_by_ct = pd.DataFrame(fc_by_ct,
                            index=grped.loc[ctest_group].index,
                            columns=adata.var_names)
    return fc_by_ct


def gene_fc_heatmap(adata: anndata.AnnData,
                    groupby: str,
                    contrast: str,
                    contrast_groups: tuple,
                    genes: Union[list, tuple],
                    cmap=sns.diverging_palette(150, 275, s=80, l=55, n=30),
                    figsize: tuple = (8, 8),
                    **kwargs) -> matplotlib.axes.Axes:
    '''Plot a [Groups, Genes] heatmap where each cell is valued by the 
    log2FC between two groups of a contrast.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] object.
    groupby : str
        categorical variable in `adata.obs` for grouping.
    contrast : str
        categorical variable in `adata.obs` for contrast.
    contrast_groups : tuple
        (group0, group1) in `adata.obs[contrast]` for log-fold change
        comparison. values are `log2(group0/group1)`.
    genes : list, tuple
        gene names in `adata.var_names` for plotting.
    cmap : seaborn.palettes._ColorPalette
        color palette for plot colors, compatible with 
        `seaborn.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes
    '''
    genes_in_adata = np.array(
        [True if x in adata.var_names else False for x in genes])
    dropped = np.array(genes)[~genes_in_adata]
    genes = np.array(genes)[genes_in_adata].tolist()
    if len(dropped) > 0:
        print('Dropped genes not in `adata`')
        print(dropped)
    assert len(contrast_groups) == 2, \
        '`len(contrast_groups)` must be == 2 for a binary contrast'
    sub = adata[:, genes].copy()
    if type(sub.X) != np.ndarray:
        X = sub.X.toarray()
    else:
        X = sub.X

    top_groups = np.unique(sub.obs[groupby])
    all_contrast_groups = np.unique(sub.obs[contrast])

    mat = np.zeros((len(top_groups), len(genes)), dtype=np.float)
    for i, tg in enumerate(top_groups):
        tg_bidx = sub.obs[groupby] == tg
        c0_bidx = np.logical_and(
            tg_bidx, adata.obs[contrast] == contrast_groups[0])
        c1_bidx = np.logical_and(
            tg_bidx, adata.obs[contrast] == contrast_groups[1])
        c0_means = np.mean(X[c0_bidx, :], axis=0)  # genewise means
        c1_means = np.mean(X[c1_bidx, :], axis=0)
        mat[i, :] = np.log2(c0_means / c1_means)
    mat[np.isinf(mat)] = None

    pl_df = pd.DataFrame(mat, index=top_groups, columns=genes)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(data=pl_df,
                xticklabels=True, yticklabels=True,
                cmap=cmap,
                ax=ax,
                center=0.,
                cbar_kws={'label': r'$\log_2$FC(%s/%s)' % contrast_groups},
                **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    return ax, mat
