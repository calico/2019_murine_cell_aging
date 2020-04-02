# Murine aging cell atlas

This repository contains utilities used in the analysis of the [Murine Aging Cell Atlas](http://mca.research.calicolabs.com).

## Table of Contents

* `aging_vec.py` contains functions to compute vectors of difference within an embedding between two groups of cells.
* `cell_subtype_heirarchy.py` formalizes our notions of cell states within each cell type and defines lists of marker genes gathered from the *Tabula Muris*.
* `diffex.py` contains tools for differential expression.
* `ot_distances.py` implements a discrete optimal transport distance calculation which we used to compare young and old cell types.
* `scanpy_fxn.py` contains assorted functions for quality control and clustering.
* `variation.py` contains functions for computing metrics of gene expression variance and cell-cell heterogeneity.
