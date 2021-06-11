import numpy as np
from typing import Union

#%% Stochastic reaction networks for single-gene dynamics
class ThreeStateSingleGene:
    """CME model to fit the marginal distributions of mRNA species."""

    STOICH_MAT = np.array(
        [
            [-1, 1, 0, 0],  # G0 -> G1
            [1, -1, 0, 0],  # G1 -> G0
            [0, -1, 1, 0],  # G1 -> G2
            [0, 1, -1, 0],  # G2 -> G1
            [0, 0, 0, 1],  # G0 -> RNA
            [0, 0, 0, 1],  # G1 -> RNA
            [0, 0, 0, 1],  # G2 -> RNA
            [0, 0, 0, -1],  # RNA -> 0
        ],
        dtype=np.intc,
    )

    NUM_REACTIONS = STOICH_MAT.shape[0]
    NUM_SPECIES = STOICH_MAT.shape[1]

    X0 = np.array([[1, 0, 0, 0]])
    P0 = np.array([1.0])
    INIT_BOUNDS = np.array([2, 2, 2, 15])

    T0 = 24 * 3600

    T_MEASUREMENTS = np.array([0, 0.5, 1, 2, 4]) * 3600

    NUM_PARAMETERS = 11

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
            out[:] = states[:, 1]
        elif reaction == 3:
            out[:] = states[:, 2]
        elif reaction == 4:
            out[:] = states[:, 0]
        elif reaction == 5:
            out[:] = states[:, 1]
        elif reaction == 6:
            out[:] = states[:, 2]
        elif reaction == 7:
            out[:] = states[:, 3]
        i_out_of_bound = np.where(states[:, 3] >= 5000)
        out[i_out_of_bound] = 0.0

    @classmethod
    def get_parameter_bounds(cls):

        PAR_LB = np.array([v for k, v in cls.LOWER_BOUND.items()])
        PAR_UB = np.array([v for k, v in cls.UPPER_BOUND.items()])

        return PAR_LB, PAR_UB


class ThreeStateSingleGeneA(ThreeStateSingleGene):
    """CME model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 11

    PARAMETERS = [
        "r1",
        "r2",
        "k01a",
        "k01b",
        "k10",
        "k12",
        "k21",
        "alpha0",
        "alpha1",
        "alpha2",
        "gamma",
    ]

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01a": 2,
        "k01b": 3,
        "k10": 4,
        "k12": 5,
        "k21": 6,
        "alpha0": 7,
        "alpha1": 8,
        "alpha2": 9,
        "gamma": 10,
    }

    LOWER_BOUND = {
        "r1": 1.0e-4,
        "r2": 1.0e-4,
        "k01a": 1.0e-4,
        "k01b": 1.0e-9,
        "k10": 1.0e-4,
        "k12": 1.0e-9,
        "k21": 1.0 / (4 * 3600),
        "alpha0": 1.0e-8,
        "alpha1": 1.0e-6,
        "alpha2": 1.0 / 3600,
        "gamma": 1.0 / (10 * 3600),
    }

    UPPER_BOUND = {
        "r1": 1.0e-2,
        "r2": 1.0e-2,
        "k01a": 10.0,
        "k01b": 500,
        "k10": 500.0,
        "k12": 10.0,
        "k21": 10.0,
        "alpha0": 1.0,
        "alpha1": 1.0,
        "alpha2": 1.0,
        "gamma": 1 / (30 * 60),
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
        out[2] = self.k12
        out[3] = self.k21
        out[4] = self.alpha0
        out[5] = self.alpha1
        out[6] = self.alpha2
        out[7] = self.gamma


class ThreeStateSingleGeneB(ThreeStateSingleGene):
    """CME model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 11

    PARAMETERS = [
        "r1",
        "r2",
        "k01",
        "k10a",
        "k10b",
        "k12",
        "k21",
        "alpha0",
        "alpha1",
        "alpha2",
        "gamma",
    ]

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01": 2,
        "k10a": 3,
        "k10b": 4,
        "k12": 5,
        "k21": 6,
        "alpha0": 7,
        "alpha1": 8,
        "alpha2": 9,
        "gamma": 10,
    }

    LOWER_BOUND = {
        "r1": 1.0e-4,
        "r2": 1.0e-4,
        "k01": 1.0e-4,
        "k10a": 1.0e-4,
        "k10b": 1.0e-9,
        "k12": 1.0e-9,
        "k21": 1.0 / (4 * 3600),
        "alpha0": 1.0e-8,
        "alpha1": 1.0e-6,
        "alpha2": 1.0 / 3600,
        "gamma": 1.0 / (10 * 3600),
    }

    UPPER_BOUND = {
        "r1": 1.0e-2,
        "r2": 1.0e-2,
        "k01": 10.0,
        "k10a": 5000,
        "k10b": 5000.0,
        "k12": 10.0,
        "k21": 10.0,
        "alpha0": 1.0,
        "alpha1": 1.0,
        "alpha2": 1.0,
        "gamma": 1 / (30 * 60),
    }

    PARAMETER_MEANINGS = {
        "r1": r"NF-$\kappa$B signal parameter",
        "r2": r"NF-$\kappa$B signal parameter",
        "k01": "Gene activation rate (events/second)",
        "k10a": "Gene switching off rate (events/second)",
        "k10b": "NF-$\kappa$B-proportional decrease in gene switching off rate (events/second)",
        "k12": "Gene state transition rate from active to highly active (events/second)",
        "k21": "Gene state transition rate from highly active to active (events/second)",
        "alpha0": "Basal mRNA production rate (molecules/second)",
        "alpha1": "mRNA production rate when gene is active (molecules/second)",
        "alpha2": "mRNA production rate when gene is highly active (molecules/second)",
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
        out[2] = self.k12
        out[3] = self.k21
        out[4] = self.alpha0
        out[5] = self.alpha1
        out[6] = self.alpha2
        out[7] = self.gamma


class ThreeStateSingleGeneC(ThreeStateSingleGene):
    """CME model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 11

    PARAMETERS = [
        "r1",
        "r2",
        "k01",
        "k10",
        "k12a",
        "k12b",
        "k21",
        "alpha0",
        "alpha1",
        "alpha2",
        "gamma",
    ]

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01": 2,
        "k10": 3,
        "k12a": 4,
        "k12b": 5,
        "k21": 6,
        "alpha0": 7,
        "alpha1": 8,
        "alpha2": 9,
        "gamma": 10,
    }

    LOWER_BOUND = {
        "r1": 1.0e-4,
        "r2": 1.0e-4,
        "k01": 1.0e-4,
        "k10": 1.0e-4,
        "k12a": 1.0e-9,
        "k12b": 1.0e-9,
        "k21": 1.0 / (4 * 3600),
        "alpha0": 1.0e-8,
        "alpha1": 1.0e-6,
        "alpha2": 1.0 / 3600,
        "gamma": 1.0 / (10 * 3600),
    }

    UPPER_BOUND = {
        "r1": 1.0e-2,
        "r2": 1.0e-2,
        "k01": 10.0,
        "k10": 500.0,
        "k12a": 10.0,
        "k12b": 500,
        "k21": 10.0,
        "alpha0": 1.0,
        "alpha1": 1.0,
        "alpha2": 1.0,
        "gamma": 1 / (30 * 60),
    }

    PARAMETER_MEANINGS = {
        "r1": r"NF-$\kappa$B signal parameter",
        "r2": r"NF-$\kappa$B signal parameter",
        "k01": "Gene activation rate (events/second)",
        "k10": "gene switching off rate (events/second)",
        "k12a": "Gene state transition rate from active to highly active (events/second)",
        "k12b": "NF-$\kappa$B-proportional increase in activation rate (events/second)",
        "k21": "Gene state transition rate from highly active to active (events/second)",
        "alpha0": "Basal mRNA production rate (molecules/second)",
        "alpha1": "mRNA production rate when gene is active (molecules/second)",
        "alpha2": "mRNA production rate when gene is highly active (molecules/second)",
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
        out[1] = self.k10
        out[2] = self.k12a + self.k12b * signal
        out[3] = self.k21
        out[4] = self.alpha0
        out[5] = self.alpha1
        out[6] = self.alpha2
        out[7] = self.gamma


class ThreeStateSingleGeneD(ThreeStateSingleGene):
    """CME model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 11

    PARAMETER = [
        "r1",
        "r2",
        "k01",
        "k10",
        "k12",
        "k21a",
        "k21b",
        "alpha0",
        "alpha1",
        "alpha2",
        "gamma",
    ]

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01": 2,
        "k10": 3,
        "k12": 4,
        "k21a": 5,
        "k21b": 6,
        "alpha0": 7,
        "alpha1": 8,
        "alpha2": 9,
        "gamma": 10,
    }

    LOWER_BOUND = {
        "r1": 1.0e-4,
        "r2": 1.0e-4,
        "k01": 1.0e-4,
        "k10": 1.0e-4,
        "k12": 1.0e-9,
        "k21a": 1.0 / (4 * 3600),
        "k21b": 1.0e-9,
        "alpha0": 1.0e-8,
        "alpha1": 1.0e-6,
        "alpha2": 1.0 / 3600,
        "gamma": 1.0 / (10 * 3600),
    }

    UPPER_BOUND = {
        "r1": 1.0e-2,
        "r2": 1.0e-2,
        "k01": 10.0,
        "k10": 5000,
        "k12": 10.0,
        "k21a": 10.0,
        "k21b": 5000.0,
        "alpha0": 1.0,
        "alpha1": 1.0,
        "alpha2": 1.0,
        "gamma": 1 / (30 * 60),
    }

    PARAMETER_MEANINGS = {
        "r1": r"NF-$\kappa$B signal parameter",
        "r2": r"NF-$\kappa$B signal parameter",
        "k01": "Gene activation rate (events/second)",
        "k10": "Gene switching off rate (events/second)",
        "k12": "Gene state transition rate from active to highly active (events/second)",
        "k21a": "Signal-independent gene state transition rate from highly active to active (events/second)",
        "k21b": "NF-$\kappa$B-proportional decrease in gene switching off rate (events/second)",
        "alpha0": "Basal mRNA production rate (molecules/second)",
        "alpha1": "mRNA production rate when gene is active (molecules/second)",
        "alpha2": "mRNA production rate when gene is highly active (molecules/second)",
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
        out[1] = self.k10
        out[2] = self.k12
        out[3] = max(0.0, self.k21a - self.k21b * signal)
        out[4] = self.alpha0
        out[5] = self.alpha1
        out[6] = self.alpha2
        out[7] = self.gamma

#%% Stochastic reaction networks for circular single-gene dynamics
class CircularThreeStateSingleGene:
    """CME model to fit the marginal distributions of mRNA species."""

    STOICH_MAT = np.array(
        [
            [-1, 1, 0, 0],  # G0 -> G1
            [1, -1, 0, 0],  # G1 -> G0
            [0, -1, 1, 0],  # G1 -> G2
            [0, 1, -1, 0],  # G2 -> G1
            [1, 0, -1, 0], # G2 -> G0
            [-1, 0, 1, 0], # G0 -> G2
            [0, 0, 0, 1],  # G0 -> RNA
            [0, 0, 0, 1],  # G1 -> RNA
            [0, 0, 0, 1],  # G2 -> RNA
            [0, 0, 0, -1],  # RNA -> 0
        ],
        dtype=np.intc,
    )

    NUM_REACTIONS = STOICH_MAT.shape[0]
    NUM_SPECIES = STOICH_MAT.shape[1]

    X0 = np.array([[1, 0, 0, 0]])
    P0 = np.array([1.0])
    INIT_BOUNDS = np.array([2, 2, 2, 15])

    T0 = 24 * 3600

    T_MEASUREMENTS = np.array([0, 0.5, 1, 2, 4]) * 3600

    NUM_PARAMETERS = 13

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
            out[:] = states[:, 1]
        elif reaction == 3:
            out[:] = states[:, 2]
        elif reaction == 4:
            out[:] = states[:, 2]
        elif reaction == 5:
            out[:] = states[:, 0]
        elif reaction == 6:
            out[:] = states[:, 0]
        elif reaction == 7:
            out[:] = states[:, 1]
        elif reaction == 8:
            out[:] = states[:, 2]
        elif reaction == 9:
            out[:] = states[:, 3]
        i_out_of_bound = np.where(states[:, 3] >= 5000)
        out[i_out_of_bound] = 0.0

    @classmethod
    def get_parameter_bounds(cls):

        PAR_LB = np.array([v for k, v in cls.LOWER_BOUND.items()])
        PAR_UB = np.array([v for k, v in cls.UPPER_BOUND.items()])

        return PAR_LB, PAR_UB

class CircularThreeStateSingleGeneA(CircularThreeStateSingleGene):
    """CME model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 13

    PARAMETERS = [
        "r1",
        "r2",
        "k01a",
        "k01b",
        "k10",
        "k12",
        "k21",
        "k20",
        "k02",
        "alpha0",
        "alpha1",
        "alpha2",
        "gamma",
    ]

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01a": 2,
        "k01b": 3,
        "k10": 4,
        "k12": 5,
        "k21": 6,
        "k20": 7,
        "k02": 8,
        "alpha0": 9,
        "alpha1": 10,
        "alpha2": 11,
        "gamma": 12,
    }

    LOWER_BOUND = {
        "r1": 1.0e-4,
        "r2": 1.0e-4,
        "k01a": 1.0e-9,
        "k01b": 1.0e-9,
        "k10": 1.0e-9,
        "k12": 1.0e-9,
        "k21": 1.0e-9,
        "k20": 1.0e-9,
        "k02": 1.0e-9,
        "alpha0": 1.0e-8,
        "alpha1": 1.0e-6,
        "alpha2": 1.0 / 3600,
        "gamma": 1.0 / (10 * 3600),
    }

    UPPER_BOUND = {
        "r1": 1.0e-2,
        "r2": 1.0e-2,
        "k01a": 1000.0,
        "k01b": 5000,
        "k10": 5000.0,
        "k12": 1000.0,
        "k21": 1000.0,
        "k20": 1000.0,
        "k02": 1000.0,
        "alpha0": 1.0,
        "alpha1": 1.0,
        "alpha2": 1.0,
        "gamma": 1 / (30 * 60),
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
        out[2] = self.k12
        out[3] = self.k21
        out[4] = self.k20
        out[5] = self.k02
        out[6] = self.alpha0
        out[7] = self.alpha1
        out[8] = self.alpha2
        out[9] = self.gamma


class CircularThreeStateSingleGeneB(CircularThreeStateSingleGene):
    """CME model to fit the marginal distributions of mRNA species."""

    NUM_PARAMETERS = 13

    PARAMETERS = [
        "r1",
        "r2",
        "k01",
        "k10a",
        "k10b",
        "k12",
        "k21",
        "k20",
        "k02",
        "alpha0",
        "alpha1",
        "alpha2",
        "gamma",
    ]

    PARAMETER_INDICES = {
        "r1": 0,
        "r2": 1,
        "k01": 2,
        "k10a": 3,
        "k10b": 4,
        "k12": 5,
        "k21": 6,
        "k20": 7,
        "k02": 8,
        "alpha0": 9,
        "alpha1": 10,
        "alpha2": 11,
        "gamma": 12,
    }

    LOWER_BOUND = {
        "r1": 1.0e-4,
        "r2": 1.0e-4,
        "k01": 1.0e-9,
        "k10a": 1.0e-9,
        "k10b": 1.0e-9,
        "k12": 1.0e-9,
        "k21": 1.0 / (4 * 3600),
        "k20": 1.0e-9,
        "k02": 1.0e-9,
        "alpha0": 1.0e-8,
        "alpha1": 1.0e-6,
        "alpha2": 1.0 / 3600,
        "gamma": 1.0 / (10 * 3600),
    }

    UPPER_BOUND = {
        "r1": 1.0e-2,
        "r2": 1.0e-2,
        "k01": 1000.0,
        "k10a": 5000.0,
        "k10b": 5000.0,
        "k12": 1000.0,
        "k21": 1000.0,
        "k20": 1000.0,
        "k02": 1000.0,
        "alpha0": 1.0,
        "alpha1": 1.0,
        "alpha2": 1.0,
        "gamma": 1 / (30 * 60),
    }

    PARAMETER_MEANINGS = {
        "r1": r"NF-$\kappa$B signal parameter",
        "r2": r"NF-$\kappa$B signal parameter",
        "k01": "Gene activation rate (events/second)",
        "k10a": "Gene switching off rate (events/second)",
        "k10b": "NF-$\kappa$B-proportional decrease in gene switching off rate (events/second)",
        "k12": "Gene state transition rate from active to highly active (events/second)",
        "k21": "Gene state transition rate from highly active to active (events/second)",
        "alpha0": "Basal mRNA production rate (molecules/second)",
        "alpha1": "mRNA production rate when gene is active (molecules/second)",
        "alpha2": "mRNA production rate when gene is highly active (molecules/second)",
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
        out[2] = self.k12
        out[3] = self.k21
        out[4] = self.k20
        out[5] = self.k02
        out[6] = self.alpha0
        out[7] = self.alpha1
        out[8] = self.alpha2
        out[9] = self.gamma