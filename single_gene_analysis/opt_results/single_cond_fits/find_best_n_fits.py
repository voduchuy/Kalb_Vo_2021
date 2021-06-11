import sys
sys.path.append("../../")
sys.path.append("../../../")
from fit_settings import *
from pathlib import Path
from models import (
    TwoStateSingleGeneA, TwoStateSingleGeneB,
    ThreeStateSingleGeneA,
    ThreeStateSingleGeneB,
    ThreeStateSingleGeneC,
    ThreeStateSingleGeneD,
    CircularThreeStateSingleGeneA,
    CircularThreeStateSingleGeneB
)

#%%
OPTIONS = {
    "fit_problem": "3SA_il1b_NoInhibitors",
    "num_select": 100,
    "fit_dir": ".",
    "output_dir": ".",
}
#%%
ARGV = sys.argv

for i in range(1, len(ARGV)):
    key, value = ARGV[i].split("=")
    if key in OPTIONS:
        OPTIONS[key] = value
    else:
        print(f"WARNING: Unknown option {key} \n")
FIT_DIR = OPTIONS["fit_dir"]
FIT_PROBLEM = OPTIONS["fit_problem"]
NUM_SELECT = int(OPTIONS["num_select"])
OUTPUT_DIR = OPTIONS["output_dir"]
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
elif MODEL_CLASS == "C3SA":
    MODEL = CircularThreeStateSingleGeneA
    RNA_IDX = 3
elif MODEL_CLASS == "C3SB":
    MODEL = CircularThreeStateSingleGeneB
    RNA_IDX = 3
#%%
if __name__ == "__main__":
    p = Path(FIT_DIR)
    result_files = list(p.glob(f'**/{FIT_PROBLEM}*.npz'))

    all_dvs = []
    all_fs = []
    for path in result_files:
        with np.load(path) as file:
            if "dv" in file: # Fit from SA or another "linear chain" algorithm
                all_dvs.append(np.squeeze(file["dv"]))
                all_fs.append(np.squeeze(file["fitness"]))
            else:
                all_dvs.append(np.squeeze(file["xs"]))
                all_fs.append(np.squeeze(file["fs"]))

    all_dvs = np.concatenate(all_dvs, axis=0)
    all_fs = np.concatenate(all_fs, axis=0)

    idx = np.argsort(all_fs, axis=0)

    np.savez(f"{OUTPUT_DIR}/{FIT_PROBLEM}_best_{NUM_SELECT}_fits.npz", xs=all_dvs[idx[0:NUM_SELECT],:],
             fs=all_fs[idx[0:NUM_SELECT]])

    print(all_dvs[idx[0:NUM_SELECT]])
    print(all_fs[idx[0:NUM_SELECT]])
    #
    # fig, axs = plt.subplots(1, MODEL.NUM_PARAMETERS)
    # for p in range(MODEL.NUM_PARAMETERS):
    #     axs[p].hist(all_dvs[:, p])
    # fig.suptitle(FIT_PROBLEM)
    # plt.show()
    # plt.close(fig)
