'''Define cell type subtype hierarchy'''
import numpy as np
import torch

'''
The following hash table provides a formal description of our 
cell type::cell state ontology, which we combine to describe
unique cell identities.

We name canonical cell states (e.g. CD4 T cell) where possible,
and otherwise name transcriptionally distinct cell states based
on their most prominent marker gene (e.g. Gucy1a3 stromal cell).
'''

# names too long to type
tub_name = 'kidney proximal straight tubule epithelial cell'

# Key: Cell Type
# Value: tuple of valid subtypes
cell_types = {
    'macrophage': (
        'CD4 macrophage',
        'CD8 macrophage',
        'activated macrophage'),
    'T cell': (
        'CD4 T cell',
        'CD8 T cell',
        'memory T cell'),
    'stromal cell': (
        'Npnt stromal cell',
        'Dcn stromal cell',
        'Hhip stromal cell',
        'Gucy1a3 stromal cell',
        'Npnt stromal cell'),
    'kidney collecting duct epithelial cell': (
        'Cald1 kidney collecting duct epithelial cell',
        'Aqp3 kidney collecting duct epithelial cell',
        'Slc12a3 kidney collecting duct epithelial cell'),
    'kidney capillary endothelial cell': (
        'Pvlap kidney capillary endothelial cell',
        'Vim kidney capillary endothelial cell',
        'Ehd3 kidney capillary endothelial cell',),
    tub_name: (
        'Fabp3 %s' % tub_name,
        'Prlr %s' % tub_name,
        'Kap %s' % tub_name,),
}


def find_likely_subtype(
    cell_type_assignment: np.ndarray,
    subtype_order: np.ndarray,
    scores: np.ndarray,
    hierarchy: dict,
) -> np.ndarray:
    '''Find the most likely valid subtype for each cell using
    given subtype classifier output and a type::subtype hierarchy.

    Parameters
    ----------
    cell_type_assignment : np.ndarray
        [Cells,] cell type assignments. must match keys in `hierarchy`.
    subtype_order : np.ndarray
        [Subtypes,] subtype labels represented by each column of `scores`.
        must match values listed in `hierarchy`.
    scores : np.ndarray
        [Cells, Subtypes] scores from a classification model. 
        NOT softmax scaled. A softmax activation is applied to only 
        the scores of valid subtypes when identifying likely subtypes.
    hierarchy : dict
        keys are cell types, values are tuples of valid subtype names.
        defines which subtypes are valid for a cell type.

    Returns
    -------
    predicted_subtypes : np.ndarray
        [Cells,] subtype assignments with only valid subtypes allowed.
    '''

    uniq_cell_types = np.unique(cell_type_assignment)
    uniq_cell_types_w_subtypes = np.array(
        [x for x in uniq_cell_types if x in hierarchy.keys()])

    predicted_subtypes = np.array(cell_type_assignment).copy()

    if not subtype_order.shape[0] == scores.shape[1]:
        msg = f'{subtype_order.shape[0]} subtypes does not match {scores.shape[1]} scores.'
        raise ValueError(msg)

    for i, ct in enumerate(uniq_cell_types_w_subtypes):
        ct_bidx = cell_type_assignment == ct

        # find valid subtype columns
        valid_subtypes = hierarchy[ct]
        st_bidx = np.array(
            [True if x in valid_subtypes else False for x in subtype_order])
        ct_valid_scores = scores[ct_bidx, :][:, st_bidx]

        # softmax transform
        sm_valid_scores = torch.nn.functional.softmax(
            torch.from_numpy(ct_valid_scores),
            dim=1)
        # likeliest subtype idx
        _, pred = torch.max(sm_valid_scores, dim=1)
        # order the subtypes in the same order they appear in the scores
        subtypes_ordered = subtype_order[st_bidx]
        pred_idx = pred.detach().squeeze().numpy().astype(np.int)
        subtype_name = subtypes_ordered[pred_idx]
        predicted_subtypes[ct_bidx] = subtype_name

    return predicted_subtypes


'''Short naming scheme

These hash tables describe the mapping from official OBO cell ontology classes
to short, colloquial names we use for cell types in our paper.

`tissue_shortener` removes tissue names in contexts where the tissue can be 
assumed (e.g. a UMAP plot of the kidney can omit kidney prefixes).
'''
shortener = {'natural killer cell': 'NK cell',
             'kidney collecting duct epithelial cell': 'kidney duct epi.',
             'kidney capillary endothelial cell': 'kidney cap. endo.',
             'kidney loop of Henle ascending limb epithelial cell': 'kidney loop. epi.',
             'kidney proximal straight tubule epithelial cell': 'kidney tub. epi.',
             'ciliated columnar cell of tracheobronchial tree': 'columnar tracheo.',
             'stromal cell': 'stromal',
             'lung endothelial cell': 'lung endothelial',
             }

tissue_shortener = {
    'natural killer cell': 'NK cell',
    'kidney collecting duct epithelial cell': 'duct epi.',
    'kidney capillary endothelial cell': 'cap. endo.',
    'kidney loop of Henle ascending limb epithelial cell': 'loop. epi.',
    'kidney proximal straight tubule epithelial cell': 'tub. epi.',
    'ciliated columnar cell of tracheobronchial tree': 'ciliated tracheobronch.',
    'stromal cell': 'stromal',
    'lung endothelial cell': 'endothelial',
}


def cellonto2shortname(long_names: list, shortener: dict) -> list:
    '''Convert cell ontology classes to short names

    Parameters
    ----------
    long_names : list
        str long names in the cell ontology class hierarchy.
    shortener : dict
        keyed by strings to replace in `long_names`, values with their
        short replacements.
        i.e. 'natural killer cell' : 'NK cell'

    Returns
    -------
    names : list
        shortened list of names.
    '''
    for k in shortener:
        long_names = [x.replace(k, shortener[k]) for x in long_names]
    return long_names


'''Marker genes for cell types.
Garnered from the Tabula Muris supplement.
https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0590-4/MediaObjects/41586_2018_590_MOESM3_ESM.pdf

This hash table formally describes the marker genes we used to check out cell
type identifications against cell types in the Tabula Muris.
'''

marker_genes = {
    'T cell': ['Cd3e', 'Cd4', 'Cd8a', ],
    'B cell': ['Cd79a', 'Cd22', 'Cd19'],
    'natural killer cell': ['Nkg7', 'Klrb1c'],
    'classical monocyte': ['Cd14', 'Fcgr1'],
    'non-classical monocyte': ['Fcgr3', 'Tnfrsf1b'],
    'macrophage': ['Cd68', 'Emr1'],
    'kidney collecting duct epithelial cell': ['Slc12a3', 'Pvalb'],
    'kidney capillary endothelial cell': ['Pecam1', 'Podxl', ],
    'kidney proximal straight tubule epithelial cell': ['Vil1', 'Slc22a12'],
    'mesangial cell': ['Des', 'Gucy1a3', 'Cspg4', 'Pdgfrb', 'Acta2', 'Vim', 'Gsn', 'Dcn'],
    'kidney cell': ['Itgam'],
    'type II pneumocyte': ['Ager', 'Epcam', 'Sftpb'],
    'stromal cell': ['Npnt', 'Gucy1a3', 'Pdpn', 'Col1a1'],
    'mast cell': ['Cd200r3', ],
    'lung endothelial cell': ['Pecam1'],
    'alveolar macrophage': ['Mrc1', 'Itgax'],
    'ciliated columnar cell of tracheobronchial tree': ['Cd24a'], }
