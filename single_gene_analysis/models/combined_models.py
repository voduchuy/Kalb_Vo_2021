from typing import Union
from .three_state_models import (
    ThreeStateSingleGeneA,
    ThreeStateSingleGeneB,
    ThreeStateSingleGeneC,
    ThreeStateSingleGeneD,
)
import numpy as np

DRUG_EFFECT = {
    "MG": {"A": ["k01b"], "B": ["k10b"], "C": ["k12b"]},
    "U0126": {"A": ["k01a"], "B": ["k01"], "C": ["k01"]}
}

MODEL_CODES = {
    "A": ThreeStateSingleGeneA,
    "B": ThreeStateSingleGeneB,
    "C": ThreeStateSingleGeneC,
    "D": ThreeStateSingleGeneD
}


class CombinedModel:

    IL1B_SRN = ThreeStateSingleGeneA
    TNFA_SRN = ThreeStateSingleGeneA

    PARAMETERS = []

    PARAMETER_INDICES = {}

    COND2PAR_MAP = {
        "il1b": {"NoInhibitors": {}, "MG": {}, "U0126": {}, "MG_U0126": {}},
        "tnfa": {"NoInhibitors": {}, "MG": {}, "U0126": {}, "MG_U0126": {}},
    }

    LOWER_BOUND = {}

    UPPER_BOUND = {}

    def __init__(self, il1b_scheme: str = "A", tnfa_scheme: str = "A"):

        self.IL1B_SCHEME = il1b_scheme
        self.TNFA_SCHEME = tnfa_scheme

        self.IL1B_SRN = MODEL_CODES[il1b_scheme]
        self.TNFA_SRN = MODEL_CODES[tnfa_scheme]

        self.PARAMETERS = (
            ["r1", "r2"]
            + [f"{s}_il1b" for s in self.IL1B_SRN.PARAMETERS[2:]]
            + [f"{s}_il1b_MG" for s in DRUG_EFFECT["MG"][self.IL1B_SCHEME]]
            + [f"{s}_il1b_U0126" for s in DRUG_EFFECT["U0126"][self.IL1B_SCHEME]]
            + [f"{s}_tnfa" for s in self.TNFA_SRN.PARAMETERS[2:]]
            + [f"{s}_tnfa_MG" for s in DRUG_EFFECT["MG"][self.TNFA_SCHEME]]
            + [f"{s}_tnfa_U0126" for s in DRUG_EFFECT["U0126"][self.TNFA_SCHEME]]
        )

        self.PARAMETER_INDICES = dict(zip(self.PARAMETERS, range(len(self.PARAMETERS))))

        self.LOWER_BOUND = {
            "r1": self.IL1B_SRN.LOWER_BOUND["r1"],
            "r2": self.IL1B_SRN.LOWER_BOUND["r2"],
            **{
                f"{s}_il1b": self.IL1B_SRN.LOWER_BOUND[s]
                for s in self.IL1B_SRN.PARAMETERS[2:]
            },
            **{
                f"{s}_il1b_MG": self.IL1B_SRN.LOWER_BOUND[s]
                for s in DRUG_EFFECT["MG"][self.IL1B_SCHEME]
            },
            **{
                f"{s}_il1b_U0126": self.IL1B_SRN.LOWER_BOUND[s]
                for s in DRUG_EFFECT["U0126"][self.IL1B_SCHEME]
            },
            **{
                f"{s}_tnfa": self.TNFA_SRN.LOWER_BOUND[s]
                for s in self.TNFA_SRN.PARAMETERS[2:]
            },
            **{
                f"{s}_tnfa_MG": self.TNFA_SRN.LOWER_BOUND[s]
                for s in DRUG_EFFECT["MG"][self.TNFA_SCHEME]
            },
            **{
                f"{s}_tnfa_U0126": self.TNFA_SRN.LOWER_BOUND[s]
                for s in DRUG_EFFECT["U0126"][self.TNFA_SCHEME]
            },
        }

        self.UPPER_BOUND = {
            "r1": self.IL1B_SRN.UPPER_BOUND["r1"],
            "r2": self.IL1B_SRN.UPPER_BOUND["r2"],
            **{
                f"{s}_il1b": self.IL1B_SRN.UPPER_BOUND[s]
                for s in self.IL1B_SRN.PARAMETERS[2:]
            },
            **{
                f"{s}_il1b_MG": self.IL1B_SRN.UPPER_BOUND[s]
                for s in DRUG_EFFECT["MG"][self.IL1B_SCHEME]
            },
            **{
                f"{s}_il1b_U0126": self.IL1B_SRN.UPPER_BOUND[s]
                for s in DRUG_EFFECT["U0126"][self.IL1B_SCHEME]
            },
            **{
                f"{s}_tnfa": self.TNFA_SRN.UPPER_BOUND[s]
                for s in self.TNFA_SRN.PARAMETERS[2:]
            },
            **{
                f"{s}_tnfa_MG": self.TNFA_SRN.UPPER_BOUND[s]
                for s in DRUG_EFFECT["MG"][self.TNFA_SCHEME]
            },
            **{
                f"{s}_tnfa_U0126": self.TNFA_SRN.UPPER_BOUND[s]
                for s in DRUG_EFFECT["U0126"][self.TNFA_SCHEME]
            },
        }
        self._form_cond2par_map()

    def _form_cond2par_map(self):
        for condition in ["NoInhibitors", "MG", "U0126", "MG_U0126"]:
            self.COND2PAR_MAP["il1b"][condition] = {
                "r1": "r1",
                "r2": "r2",
                **{s: f"{s}_il1b" for s in self.IL1B_SRN.PARAMETERS[2:]},
            }
            self.COND2PAR_MAP["tnfa"][condition] = {
                "r1": "r1",
                "r2": "r2",
                **{s: f"{s}_tnfa" for s in self.TNFA_SRN.PARAMETERS[2:]},
            }

        for condition in ["MG", "U0126"]:
            for s in DRUG_EFFECT[condition][self.IL1B_SCHEME]:
                self.COND2PAR_MAP["il1b"][condition][f"{s}"] = f"{s}_il1b_{condition}"
                self.COND2PAR_MAP["il1b"]["MG_U0126"][f"{s}"] = f"{s}_il1b_{condition}"
            for s in DRUG_EFFECT[condition][self.TNFA_SCHEME]:
                self.COND2PAR_MAP["tnfa"][condition][f"{s}"] = f"{s}_tnfa_{condition}"
                self.COND2PAR_MAP["tnfa"]["MG_U0126"][f"{s}"] = f"{s}_tnfa_{condition}"

    def get_parameter_bounds(self):

        lb = np.array([v for k, v in self.LOWER_BOUND.items()])
        ub = np.array([v for k, v in self.UPPER_BOUND.items()])

        return lb, ub

    def convert_to_single_gene_parameters(
        self, theta: Union[np.ndarray, dict], species: str, condition: str
    ) -> dict:
        if isinstance(theta, np.ndarray):
            ans = {
                k: theta[self.PARAMETER_INDICES[v]]
                for k, v in self.COND2PAR_MAP[species][condition].items()
            }
        elif isinstance(theta, dict):
            ans = {
                k: theta[v] for k, v in self.COND2PAR_MAP[species][condition].items()
            }
        else:
            raise TypeError(
                "Input parameter set needs to be either a numpy array or a dict."
            )
        return ans
