import numpy as np
from typing import Union


class TwoStateSingleGene:
    """CME model to fit the marginal distributions of mRNA species."""

    STOICH_MAT = np.array(
        [
            [-1, 1, 0],  # G0 -> G1
            [1, -1, 0],  # G1 -> G0
            [0, 0, 1],  # G0 -> RNA
            [0, 0, 1],  # G1 -> RNA
            [0, 0, -1],  # RNA -> 0
        ],
        dtype=np.intc,
    )

    NUM_REACTIONS = STOICH_MAT.shape[0]
    NUM_SPECIES = STOICH_MAT.shape[1]

    X0 = np.array([[1, 0, 0]])
    P0 = np.array([1.0])
    INIT_BOUNDS = np.array([2, 2, 15])

    T0 = 24 * 3600

    T_MEASUREMENTS = np.array([0, 0.5, 1, 2, 4]) * 3600

    PARAMETER_INDICES = None

    LOWER_BOUND = None

    UPPER_BOUND = None

    def __init__(self, theta: Union[np.ndarray, dict]):
        if type(theta) == np.ndarray:
            for k, i in self.PARAMETER_INDICES.items():
                setattr(self, k, theta[i])
        elif type(theta) == dict:
            for k, v in theta.items():
                setattr(self, k, v)
        else:
            raise TypeError(
                "Input parameter set needs to be either a numpy array or a dict."
            )

    def propensity_t(self, t: float, out: np.ndarray):
        pass

    def propensity_x(self, reaction: int, states: np.ndarray, out: np.ndarray):

        if reaction == 0:
            out[:] = states[:, 0]
        elif reaction == 1:
            out[:] = states[:, 1]
        elif reaction == 2:
            out[:] = states[:, 0]
        elif reaction == 3:
            out[:] = states[:, 1]
        elif reaction == 4:
            out[:] = states[:, 2]
        i_out_of_bound = np.where(states[:, 2] >= 2500)
        out[i_out_of_bound] = 0.0

    def update_trainable_parameters(self, new_values: np.ndarray):
        pass

    @classmethod
    def get_trainable_idxs(cls):
        return np.array(range(0, cls.NUM_PARAMETERS), dtype=int)

    @classmethod
    def get_trainable_bounds(cls):

        PAR_LB = np.array([v for k, v in cls.LOWER_BOUND.items()])
        PAR_UB = np.array([v for k, v in cls.UPPER_BOUND.items()])

        return PAR_LB, PAR_UB


#%%
class TwoStateSingleGeneA(TwoStateSingleGene):
    """Two-state gene expression model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 8

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01a": 2,
        "k01b": 3,
        "k10": 4,
        "alpha0": 5,
        "alpha1": 6,
        "gamma": 7,
    }

    LOWER_BOUND = {
        "r1": 1.0e-4,
        "r2": 1.0e-4,
        "k01a": 1.0e-4,
        "k01b": 1.0e-9,
        "k10": 1.0e-4,
        "alpha0": 1.0e-8,
        "alpha1": 1.0e-6,
        "gamma": 1.0 / (10 * 3600),
    }

    UPPER_BOUND = {
        "r1": 1.0e-2,
        "r2": 1.0e-2,
        "k01a": 10.0,
        "k01b": 500,
        "k10": 500.0,
        "alpha0": 1.0,
        "alpha1": 1.0,
        "gamma": 1 / (30 * 60),
    }

    PARAMETER_MEANINGS = {
        "r1": r"NF-$\kappa$B signal parameter",
        "r2": r"NF-$\kappa$B signal parameter",
        "k01a": "Gene activation rate (events/second)",
        "k01b": "NF-$\kappa$B-proportional increase in gene switching off rate (events/second)",
        "k10": "gene switching off rate (events/second)",
        "alpha0": "Basal mRNA production rate (molecules/second)",
        "alpha1": "mRNA production rate when gene is active (molecules/second)",
        "gamma": "mRNA degradation rate (molecules/second)",
    }

    def propensity_t(self, t: float, out: np.ndarray):
        if t <= self.T0:
            signal = 0.0
        else:
            signal = np.exp(-self.r1 * (t - self.T0)) * (
                1.0 - np.exp(-self.r2 * (t - self.T0))
            )
        out[0] = self.k01a + self.k01b * signal
        out[1] = self.k10
        out[2] = self.alpha0
        out[3] = self.alpha1
        out[4] = self.gamma
#%%
class TwoStateSingleGeneB(TwoStateSingleGene):
    """Two-state gene expression model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 8

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01": 2,
        "k10a": 3,
        "k10b": 4,
        "alpha0": 5,
        "alpha1": 6,
        "gamma": 7,
    }

    LOWER_BOUND = {"r1":     1.0e-4,
     "r2":     1.0e-4,
     "k01":    1.0e-4,
     "k10a":   1.0e-4,
     "k10b":   1.0e-9,
     "alpha0": 1.0e-8,
     "alpha1": 1.0e-6,
     "gamma":  1.0 / (10 * 3600), }

    UPPER_BOUND = {"r1":     1.0e-2,
     "r2":     1.0e-2,
     "k01":    10.0,
     "k10a":   5000,
     "k10b":   5000.0,
     "alpha0": 1.0,
     "alpha1": 1.0,
     "gamma":  1 / (30 * 60), }

    PARAMETER_MEANINGS = {
        "r1": r"NF-$\kappa$B signal parameter",
        "r2": r"NF-$\kappa$B signal parameter",
        "k01": "Gene activation rate (events/second)",
        "k10a": "Gene switching off rate (events/second)",
        "k10b": "NF-$\kappa$B-proportional decrease in gene switching off rate (events/second)",
        "alpha0": "Basal mRNA production rate (molecules/second)",
        "alpha1": "mRNA production rate when gene is active (molecules/second)",
        "gamma": "mRNA degradation rate (molecules/second)",
    }

    def propensity_t(self, t: float, out: np.ndarray):
        if t <= self.T0:
            signal = 0.0
        else:
            signal = np.exp(-self.r1 * (t - self.T0)) * (
                1.0 - np.exp(-self.r2 * (t - self.T0))
            )
        out[0] = self.k01
        out[1] = max(0.0, self.k10a - self.k10b * signal)
        out[2] = self.alpha0
        out[3] = self.alpha1
        out[4] = self.gamma


