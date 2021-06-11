import sys

sys.path.append("../../../")
from two_state_models import TwoStateSingleGeneA, TwoStateSingleGeneB
from models import (
    ThreeStateSingleGeneA,
    ThreeStateSingleGeneB,
    ThreeStateSingleGeneC,
    ThreeStateSingleGeneD,
)
from fit_settings import *
import matplotlib.pyplot as plt

#%%
OPTIONS = {
    "fit_problem": "3SA_il1b_NoInhibitors",
    "population_file": None,
}
#%%
ARGV = sys.argv

for i in range(1, len(ARGV)):
    key, value = ARGV[i].split("=")
    if key in OPTIONS:
        OPTIONS[key] = value
    else:
        print(f"WARNING: Unknown option {key} \n")

FIT_PROBLEM = OPTIONS["fit_problem"]
#%%
MODEL_CLASS, SPECIES, CONDITION = FIT_PROBLEM.split("_")
if MODEL_CLASS == "3SA":
    MODEL = ThreeStateSingleGeneA
    RNA_IDX = 3
elif MODEL_CLASS == "3SB":
    MODEL = ThreeStateSingleGeneB
    RNA_IDX = 3
elif MODEL_CLASS == "3SC":
    MODEL = ThreeStateSingleGeneC
    RNA_IDX = 3
elif MODEL_CLASS == "3SD":
    MODEL = ThreeStateSingleGeneD
    RNA_IDX = 3
elif MODEL_CLASS == "2SA":
    MODEL = TwoStateSingleGeneA
    RNA_IDX = 2
elif MODEL_CLASS == "2SB":
    MODEL = TwoStateSingleGeneB
    RNA_IDX = 2
#%%
if OPTIONS["population_file"] is None:
    raise RuntimeError("Must give population file.")

with np.load(OPTIONS["population_file"], allow_pickle=True) as f:
    pop0 = [f["xs"], f["fs"]]
    POPULATION_SIZE = f["xs"].shape[0]
#%%
if __name__ == "__main__":
    LB, UB = [np.log10(b) for b in MODEL.get_parameter_bounds()]

    fit, axs = plt.subplots(MODEL.NUM_PARAMETERS, MODEL.NUM_PARAMETERS)
    for ipar in range(MODEL.NUM_PARAMETERS):
        for jpar in range(ipar + 1, MODEL.NUM_PARAMETERS):
            print(ipar, jpar)
            axs[ipar, jpar].scatter(pop0[0][:, ipar], pop0[0][:, jpar])
            axs[ipar, jpar].set_xlim([LB[ipar], UB[ipar]])
            axs[ipar, jpar].set_ylim([LB[ipar], UB[ipar]])
    plt.show()
