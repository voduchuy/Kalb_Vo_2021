# Kalb_Vo_2021

This repository contains the code used for the data modeling and analysis presented in the publication

>Daniel Kalb, Huy D Vo, Samantha Adikari, Elizabeth Hong-Geller, Brian Munsky, James Werner. Visualization and Modeling of Inhibition of IL-1β and TNFα mRNA Transcription at the Single-Cell Level. _Scientific Reports_ 11, 2021. 
>DOI: [https://doi.org/10.1038/s41598-021-92846-0](https://doi.org/10.1038/s41598-021-92846-0)

## Dependencies

For all tasks:
- Python 3.8+.

For plotting the analysis results:
- NumPy.
- Matplotlib.

For conducting the parameter fitting tasks:
- SciPy.
- [PyGMO 2](https://esa.github.io/pygmo2/).
- [MPI for Python 2+](https://mpi4py.readthedocs.io/en/stable/).
- [PACMENSL](https://github.com/voduchuy/pacmensl).
- [PyPACMESL](https://github.com/voduchuy/pypacmensl).

## Directory structure
- `data` contains the smFISH measurements of IL-1b and TNF-a transcriptions in response to LPS over 4 experimental conditions (No treatment, MG132, U0126, MG132+U0126).
- `modules` contains useful functions for loading and binning the data (`data_io`), calculating moments from the model (`moments`), generating parallel random numbers (`mpi4rng`), and routines to make PyGMO work with MPI (`mpi4pygmo`).
- `single_gene_analysis` contains scripts for the core data fitting tasks as well as Jupyter notebooks that plot the results and output them into figures and tables.
