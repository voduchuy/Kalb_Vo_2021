import sys

sys.path.append("../../")
sys.path.append("../../../")
from fit_settings import *
from pathlib import Path
from models import *

#%%
OPTIONS = {
    "model": "3SAA",
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
MODEL = OPTIONS["model"]
OUTPUT_DIR = OPTIONS["output_dir"]
#%%
if __name__ == "__main__":
    p = Path(FIT_DIR)
    print(MODEL)
    result_files = list(p.glob(f"**/{MODEL}_*.npz"))
    print(result_files)
    all_dvs = []
    all_fs = []
    for path in result_files:
        with np.load(path, allow_pickle=True) as file:
            if "par" in file and "fitness" in file:  # Fit from SA or another "linear chain" algorithm
                all_dvs.append(file["par"])
                all_fs.append(file["fitness"])

    all_dvs = np.array(all_dvs)
    all_fs = np.array(all_fs)
    print(all_dvs)
    print(all_fs)

    idx = np.argsort(all_fs, axis=0)

    np.savez(
        f"{OUTPUT_DIR}/{MODEL}_best_joint_fit.npz",
        xs=np.squeeze(all_dvs[idx[0], :]),
        fs=np.squeeze(all_fs[idx[0]]),
    )

    print(all_dvs[idx])
    print(all_fs[idx])
    #
    # fig, axs = plt.subplots(1, MODEL.NUM_PARAMETERS)
    # for p in range(MODEL.NUM_PARAMETERS):
    #     axs[p].hist(all_dvs[:, p])
    # fig.suptitle(FIT_PROBLEM)
    # plt.show()
    # plt.close(fig)
