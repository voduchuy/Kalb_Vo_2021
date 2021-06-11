import sys

sys.path.append("../../")
sys.path.append("../../../")
from fit_settings import *
from pathlib import Path
from models import (
    TwoStateSingleGeneA,
    TwoStateSingleGeneB,
    ThreeStateSingleGeneA,
    ThreeStateSingleGeneB,
    ThreeStateSingleGeneC,
    ThreeStateSingleGeneD,
    CombinedModel
)
from pathlib import Path

opts = {
        'best_fit_dir': 'bests',
        'output_dir': '.',
        'model': '3SAA'
}

for i in range(1, len(sys.argv)):
    key, value = sys.argv[i].split("=")
    if key in opts:
        opts[key] = value
    else:
        print(f"WARNING: Unknown option {key} \n")

if __name__ == "__main__":
    ALLOWED_MODELS = {
            "3SAA",
            "3SBB",
            "3SCC",
            "3SAC",
            "3SAB",
            "3SBA",
            "3SBC",
            "3SCA",
            "3SCB",
    }

    if (opt := opts["model"]) in ALLOWED_MODELS:
        combined_model = CombinedModel(opt[-2], opt[-1])
        il1b_model_code = '3S' + combined_model.IL1B_SCHEME
        tnfa_model_code = '3S' + combined_model.TNFA_SCHEME
    else:
        raise ValueError(f"Model {opts['model']} is not supported.")

    il1b_model = combined_model.IL1B_SRN
    tnfa_model = combined_model.TNFA_SRN

    LB, UB = combined_model.get_parameter_bounds()

    par = np.random.uniform(np.log10(LB), np.log10(UB))

    p = Path(opts['best_fit_dir'])

    il1b_f = list(p.glob(f'**/{il1b_model_code}_il1b_NoInhibitors*.npz'))[0]
    with np.load(il1b_f) as f:
        x = np.squeeze(f['xs'])
        for cond in ["NoInhibitors", "MG", "U0126"]:
            for k, v in combined_model.COND2PAR_MAP ['il1b'][cond].items():
                par[combined_model.PARAMETER_INDICES[v]] = x[il1b_model.PARAMETER_INDICES[k]]

    tnfa_f = list(p.glob(f'**/{tnfa_model_code}_tnfa_NoInhibitors*.npz'))[0]
    with np.load(tnfa_f) as f:
        x = np.squeeze(f['xs'])
        for cond in ["NoInhibitors", "MG", "U0126"]:
            for k, v in combined_model.COND2PAR_MAP['tnfa'][cond].items():
                par[combined_model.PARAMETER_INDICES[v]] = x[tnfa_model.PARAMETER_INDICES[k]]

    np.savez(f'{opts["output_dir"]}/{opts["model"]}_start.npz', par=par)


