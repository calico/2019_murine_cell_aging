'''Compute vectors of difference across a contrast'''
import anndata
import numpy as np
import tqdm


def compute_displacement_vector(adata: anndata.AnnData,
                                contrast: str = 'age',
                                groupby: str = 'cell_type',
                                embedding: str = 'counts') -> np.ndarray:
    '''
    Compute the difference between mean vectors for cells in 
    different groups of a binary `contrast`.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] AnnData object.
    contrast : str
        binary variable in `adata.obs`. differences in mean 
        expression vectors are computed across the two groups
        specified in this variable.
    groupby : str
        categorical variable in `adata.obs`. mean expression 
        vectors are computed separately for each level in this 
        variable.
    embedding : str
        specifies the embedding space to calculate mean vectors.
        ["counts", "pca", "umap", "nmf"]

    Returns
    -------
    difference_vectors : np.ndarray
        [n_groups, n_embedding_dim, (contrast_0, contrast_1, difference)]
        mean expression vectors for each group, contrast combination 
        and their differences.
    '''
    # [Cells, Genes]
    if embedding.lower() == 'counts':
        if type(adata.X) != np.ndarray:
            X = adata.X.toarray()
        else:
            X = adata.X
    elif embedding.lower() == 'pca':
        if 'X_pca' not in adata.obsm.keys():
            raise ValueError('PCA object not present in AnnData.')
        X = adata.obsm['X_pca']
    elif embedding.lower() == 'umap':
        if 'X_umap' not in adata.obsm.keys():
            raise ValueError('UMAP object not present in AnnData.')
        X = adata.obsm['X_umap']
    elif embedding.lower() == 'nmf':
        if 'X_nmf' not in adata.obsm.keys():
            raise ValueError('NMF object not present in AnnData.')
        X = adata.obsm['X_nmf']
    else:
        raise ValueError('invalid embedding argument')
    print('%d cells and %d features.' % X.shape)

    contrast_groups = np.unique(adata.obs[contrast])
    if len(contrast_groups) != 2:
        msg = f'`constrast` must have binary values, not {len(contrast_groups)}'
        raise ValueError(msg)
    contrast_bindices = []
    for g in contrast_groups:
        contrast_bindices.append(
            adata.obs[contrast] == g
        )

    groupby_groups = np.unique(adata.obs[groupby])
    groupby_bindices = []
    for g in groupby_groups:
        groupby_bindices.append(
            adata.obs[groupby] == g
        )

    # build matrix of difference vectors
    difference_vectors = np.zeros(
        (len(groupby_groups), X.shape[1], len(contrast_groups)+1)
    )

    for i, group in tqdm.tqdm(enumerate(groupby_groups),
                              desc='Computing group contrasts'):

        for j, contrast in enumerate(contrast_groups):

            group_bidx = groupby_bindices[i]
            contrast_bidx = contrast_bindices[j]
            bidx = np.logical_and(group_bidx, contrast_bidx)
            cells = X[bidx, :]

            difference_vectors[i, :, j] = cells.mean(axis=0)

        # compute difference vector
        difference_vectors[i, :, 2] = difference_vectors[i,
                                                         :, 0] - difference_vectors[i, :, 1]
    return difference_vectors
