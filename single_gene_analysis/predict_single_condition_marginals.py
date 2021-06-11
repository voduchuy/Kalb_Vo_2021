import mpi4py.MPI as mpi
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../../models")
sys.path.append("../modules")
print(sys.path)

from single_gene_analysis.models import (
TwoStateSingleGeneA, TwoStateSingleGeneB,
    ThreeStateSingleGeneA,
    ThreeStateSingleGeneB,
    ThreeStateSingleGeneC,
    ThreeStateSingleGeneD,
    CircularThreeStateSingleGeneA,
    CircularThreeStateSingleGeneB
)
from pypacmensl.fsp_solver import FspSolverMultiSinks

MY_RANK = mpi.COMM_WORLD.Get_rank()
NUM_CPUS = mpi.COMM_WORLD.Get_size()
# %%
OPTIONS = {
        'fit_file':    None,
        'output_file': 'marginal.npz',
        'model_class': '3SA'
}
# %%
argv = sys.argv
for opt in argv[1:]:
    key, val = opt.split("=")
    if key in OPTIONS:
        OPTIONS[key] = val
    else:
        print(f"WARNING: Unknown option {key} \n")
FIT_FILE = OPTIONS['fit_file']
MODEL_CLASS = OPTIONS['model_class']
# %%
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
# %%
T_MEASUREMENTS = np.array([0, 0.5, 1, 2, 4]) * 3600

tnodes = np.concatenate((T_MEASUREMENTS, np.linspace(0, 4 * 3600, 100)))
tnodes = np.unique(tnodes) + MODEL.T0


def solve_model(log10_theta):
    theta = np.power(10.0, log10_theta)
    model = MODEL(theta)
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
    solutions = SOLVER0.SolveTspan(tnodes, 1.0E-7)
    SOLVER0.ClearState()
    return solutions


def get_marginals(tnodes, solutions):
    its = np.where(np.isin(tnodes, T_MEASUREMENTS + MODEL.T0))
    its = its[0]
    marginals = []
    for it in its:
        marginals.append(solutions[it].Marginal(RNA_IDX))
    return marginals


def getModelMeanVariances(solutions):
    means = np.zeros((len(tnodes, )), dtype=float)
    variances = np.copy(means)
    for i in range(0, len(tnodes)):
        p = solutions[i].Marginal(RNA_IDX)
        means[i] = np.dot(np.arange(0, len(p)), p)
        variances[i] = np.dot(np.arange(0, len(p)) ** 2, p) - means[i] * means[i]
    return means, variances

#%%
if __name__ == '__main__':

    with np.load(FIT_FILE) as file:
        if 'par' in file:
            log10_theta = file['par']
        elif 'xs' in file:
            xs = file['xs']
            fs = file['fs']
            ibest = np.argmax(-fs)
            log10_theta = xs[ibest,:]

    print(log10_theta)
    solutions = solve_model(log10_theta)

    marginals = get_marginals(tnodes, solutions)
    means, vars = getModelMeanVariances(solutions)

    np.savez(OPTIONS['output_file'], tnodes=tnodes, mean=means, var=vars, p0=marginals[0], p1=marginals[1],
             p2=marginals[2],
             p3=marginals[3], p4=marginals[4])
