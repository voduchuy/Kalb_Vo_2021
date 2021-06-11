import sys
sys.path.append("../../..")
import numpy as np
from scipy.io import loadmat
import os
import pathlib

MODULE_PATH = pathlib.PurePath(__file__)
DATA_PATH = MODULE_PATH.parents[2] / 'data'

CONDITIONS = ["NoInhibitors", "MG", "U0126", "MG_U0126"]
TIME_LABELS = ["0hr", "30mins", "1hr", "2hr", "4hr"]
SPECIES_INDICES = {"tnfa": 1, "il1b": 0}
REPLICA_LABELS = ["_0", "_1", "_2", "_3", ""]
NUM_REPLICAS = 4
NUM_TIMES = 5

def load_single_species_data(condition: str, species: str, replica=-1):
    if species.lower() == "tnfa":
        ispec = 1
    else:
        ispec = 0

    if replica == -1:
        repstr = ""
    else:
        repstr = f"_{str(replica)}"

    # 1st row IL1beta, 2nd row TNFalpha
    data = loadmat(DATA_PATH / "FinalDataAll.mat")
    obsr_tags = ["0hr", "30mins", "1hr", "2hr", "4hr"]
    data_snapshots = []
    for i in range(0, len(obsr_tags)):
        X = data["outputs_" + obsr_tags[i] + "_" + condition + repstr]
        X = np.ascontiguousarray(X[ispec, :])
        data_snapshots.append(X)
    return data_snapshots


def load_data_correlations():
    matdata = loadmat(DATA_PATH / "FinalDataAll.mat")

    all_correlations = dict()
    for cond in CONDITIONS:
        all_correlations[cond] = dict()
        for rep in REPLICA_LABELS:
            all_correlations[cond][rep] = dict()
            for t in TIME_LABELS:
                all_correlations[cond][rep][t] = np.corrcoef(
                    matdata[f"outputs_{t}_{cond}{rep}"][0:2, :], rowvar=True
                )

    matdata.clear()
    return all_correlations


def load_all_datasets():
    """
    Load all datasets from .mat file.

    Returns
    -------

    all_data: dict
        all_data[cond][rep][timepoint] stores single-cell measurements in condtion 'cond', replica 'rep',
        and timepoint 'timepoint'.
        cond is a member of ['NoInhibitors', 'MG', 'U0126', 'MG_U0126']
        timepoint is a member of ['0hr', '30mins', '1hr', '2hr', '4hr']
        rep is a member of ['_0', '_1', '_2', '_3', '']
    """
    matdata = loadmat(DATA_PATH / "FinalDataAll.mat")

    all_data = dict()
    for cond in CONDITIONS:
        all_data[cond] = dict()
        for rep in REPLICA_LABELS:
            all_data[cond][rep] = dict()
            for t in TIME_LABELS:
                all_data[cond][rep][t] = matdata[f"outputs_{t}_{cond}{rep}"][
                    0:2, :
                ]

    matdata.clear()
    return all_data

def load_data_marginals():
    NUM_TIMES = 5
    DATA_MARGINALS = {}

    for condition in ["NoInhibitors", "MG", "U0126", "MG_U0126"]:
        DATA_MARGINALS[condition] = {}
        for species in ["tnfa", "il1b"]:
            DATA_MARGINALS[condition][species] = []
            for k in range(0, NUM_TIMES):
                time = TIME_LABELS[k]
                num_bins = (
                    np.max(
                            ALL_DATA["NoInhibitors"][""][time][SPECIES_INDICES[species], :]
                    )
                    + 1
                )
                h, _ = np.histogram(
                        ALL_DATA["NoInhibitors"][""][time][SPECIES_INDICES[species], :],
                    num_bins,
                    density=True,
                )
                DATA_MARGINALS[condition][species].append(h)
    return DATA_MARGINALS

def load_data_means():
    DATA_MEANS = {}
    for condition in CONDITIONS:
        DATA_MEANS[condition] = {}
        for species in ["tnfa", "il1b"]:
            DATA_MEANS[condition][species] = np.zeros((NUM_REPLICAS + 1, NUM_TIMES))
            for j in range(0, NUM_REPLICAS):
                for k in range(0, NUM_TIMES):
                    replica = REPLICA_LABELS[j]
                    time = TIME_LABELS[k]
                    DATA_MEANS[condition][species][j, k] = np.mean(
                            ALL_DATA[condition][replica][time][
                            SPECIES_INDICES[species], :
                            ]
                    )
            for k in range(0, NUM_TIMES):
                time = TIME_LABELS[k]
                DATA_MEANS[condition][species][-1, k] = np.mean(
                        ALL_DATA[condition][""][time][SPECIES_INDICES[species], :])
    return DATA_MEANS


def load_data_covariances():
    DATA_COVARIANCES = {}

    for condition in ["NoInhibitors", "MG", "U0126", "MG_U0126"]:
        DATA_COVARIANCES[condition] = np.zeros((NUM_REPLICAS + 1, NUM_TIMES, 2, 2))
        for j in range(0, NUM_REPLICAS):
            for k in range(0, NUM_TIMES):
                replica = REPLICA_LABELS[j]
                time = TIME_LABELS[k]
                DATA_COVARIANCES[condition][j, k, :, :] = np.cov(
                        ALL_DATA[condition][replica][time]
                )
        for k in range(0, NUM_TIMES):
            replica = ""
            time = TIME_LABELS[k]
            DATA_COVARIANCES[condition][-1, k, :, :] = np.cov(
                    ALL_DATA[condition][replica][time]
            )
    return DATA_COVARIANCES

ALL_DATA = load_all_datasets()
# Count number of measurements
NUM_MEASUREMENTS = {}
for condition in CONDITIONS:
    NUM_MEASUREMENTS[condition] = np.zeros((NUM_REPLICAS+1, NUM_TIMES))
    for j in list(range(0, NUM_REPLICAS))+[-1]:
        for k in range(0, NUM_TIMES):
            replica = REPLICA_LABELS[j]
            time = TIME_LABELS[k]
            NUM_MEASUREMENTS[condition][j, k] = ALL_DATA[condition][replica][time].shape[1]