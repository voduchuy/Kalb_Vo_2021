import mpi4py.MPI as mpi
import numpy as np
import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../../models")
sys.path.append("../modules")
print(sys.path)

from typing import Union

from single_gene_analysis.models import (
    TwoStateSingleGeneA,
    TwoStateSingleGeneB,
    ThreeStateSingleGeneA,
    ThreeStateSingleGeneB,
    ThreeStateSingleGeneC,
    ThreeStateSingleGeneD,
    CombinedModel,
)
from pypacmensl.fsp_solver import FspSolverMultiSinks

MY_RANK = mpi.COMM_WORLD.Get_rank()
NUM_CPUS = mpi.COMM_WORLD.Get_size()
# %%
OPTIONS = {"fit_file": None, "output_file": "marginal.npz", "model": "3SAA"}
# %%
argv = sys.argv
for opt in argv[1:]:
    key, val = opt.split("=")
    if key in OPTIONS:
        OPTIONS[key] = val
    else:
        print(f"WARNING: Unknown option {key} \n")
FIT_FILE = OPTIONS["fit_file"]
# %%
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

if (opt := OPTIONS["model"]) in ALLOWED_MODELS:
    combined_model = CombinedModel(opt[-2], opt[-1])
    RNA_IDX = 3
else:
    raise ValueError(f"Model {OPTIONS['model']} is not supported.")
# %%
def solve_single_gene_cme(theta: Union[dict, np.ndarray], model, tnodes: np.ndarray):
    model = model(theta)
    propensity_t = model.propensity_t
    propensity_x = model.propensity_x
    SOLVER0 = FspSolverMultiSinks(mpi.COMM_WORLD)
    SOLVER0.SetVerbosity(2)
    SOLVER0.SetLBMethod("block")
    SOLVER0.SetModel(model.STOICH_MAT, propensity_t, propensity_x)
    SOLVER0.SetFspShape(constr_fun=None, constr_bound=model.INIT_BOUNDS)
    SOLVER0.SetInitialDist(model.X0, model.P0)
    SOLVER0.SetLBMethod("BLOCK")
    SOLVER0.SetOdeSolver("PETSC")
    SOLVER0.SetOdeTolerances(1.0e-4, 1.0e-8)
    SOLVER0.SetUp()
    solutions = SOLVER0.SolveTspan(tnodes + model.T0, 1.0e-7)
    SOLVER0.ClearState()
    return solutions


def get_single_gene_marginals(tnodes, solutions):
    its = np.where(np.isin(tnodes, T_MEASUREMENTS))
    its = its[0]
    marginals = []
    for it in its:
        marginals.append(solutions[it].Marginal(RNA_IDX))
    return marginals


def getModelMeanVariances(solutions):
    means = np.zeros((len(T_OUTPUTS,)), dtype=float)
    variances = np.copy(means)
    for i in range(0, len(T_OUTPUTS)):
        p = solutions[i].Marginal(RNA_IDX)
        means[i] = np.dot(np.arange(0, len(p)), p)
        variances[i] = np.dot(np.arange(0, len(p)) ** 2, p) - means[i] * means[i]
    return means, variances

def getGeneStateDistributions(solutions):
    def GeneState(x, gene):
        gene[0] = x[0]
        gene[1] = x[1]
        gene[2] = x[2]
    gene_probabilities = np.zeros((len(T_OUTPUTS), 3), dtype=float)
    for i in range(0, len(T_OUTPUTS)):
        gene_probabilities[i, :] = solutions[i].WeightedAverage(3, GeneState)
    return gene_probabilities


#%%
if __name__ == "__main__":

    with np.load(FIT_FILE) as file:
        if "par" in file:
            log10_theta = file["par"]
        elif "xs" in file:
            xs = file["xs"]
            log10_theta = np.squeeze(xs)

    T_MEASUREMENTS = np.array([0, 0.5, 1, 2, 4]) * 3600

    T_OUTPUTS = np.concatenate((T_MEASUREMENTS, np.linspace(0, 4 * 3600, 100)))
    T_OUTPUTS = np.unique(T_OUTPUTS)

    print(log10_theta)
    ans = {}
    for species in ["il1b", "tnfa"]:
        ans[species] = {}
        for condition in ["NoInhibitors", "MG", "U0126", "MG_U0126"]:
            single_gene_model = getattr(combined_model, f"{species.upper()}_SRN")
            theta_single_gene = combined_model.convert_to_single_gene_parameters(
                10.0 ** log10_theta, species, condition
            )
            print(theta_single_gene)
            solutions = solve_single_gene_cme(
                theta_single_gene, single_gene_model, T_OUTPUTS
            )
            marginals = get_single_gene_marginals(T_OUTPUTS, solutions)
            means, vars = getModelMeanVariances(solutions)
            gene_dists = getGeneStateDistributions(solutions)
            ans[species][condition] = {
                "tnodes": T_OUTPUTS,
                "mean": means,
                "var": vars,
                "gene_dists": gene_dists,
                "p0": marginals[0],
                "p1": marginals[1],
                "p2": marginals[2],
                "p3": marginals[3],
                "p4": marginals[4],
            }

    np.savez(OPTIONS["output_file"], predictions=ans, allow_pickle=True)
