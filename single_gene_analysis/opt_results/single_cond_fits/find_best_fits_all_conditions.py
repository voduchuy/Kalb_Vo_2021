import os
import sys

MODEL = ["2SA", "2SB", "3SA", "3SB", "3SC", "3SD"]
SPECIES = ["il1b", "tnfa"]
CONDS = ["NoInhibitors"]

for model in MODEL:
    for species in SPECIES:
        for cond in CONDS:
            os.system(rf"python ./find_best_n_fits.py fit_problem={model}_{species}_{cond} num_select=1 \
            fit_dir=.")