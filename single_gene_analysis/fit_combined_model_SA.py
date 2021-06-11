from typing import Union, Any
import sys

sys.path.append("../")
import mpi4py.MPI as mpi
import pygmo
import sys
from pypacmensl.fsp_solver import FspSolverMultiSinks, DiscreteDistribution
from pypacmensl.smfish.snapshot import SmFishSnapshot
from modules.data_io.data_io import load_single_species_data
from models import *
from modules.mpi4pygmo import Minimize
from fit_settings import *

#%%
ARGV = sys.argv
#%% Options for the optimization run
OPTIONS = {
    "model": "3SAC",
    "output_file": "opt_results/joint_fit.npz",
    "starting_point": None,
    "search_repeats": 1,
    "distribute_conditions": False,  # Whether to partition the available CPUs and let each subset compute
    # loss function of a single condition when evaluating the total loss function
}
#%% Parse command line arguments
for i in range(1, len(ARGV)):
    key, value = ARGV[i].split("=")
    if key in OPTIONS:
        OPTIONS[key] = value
    else:
        print(f"WARNING: Unknown option {key} \n")

OPT_REPEATS = int(OPTIONS["search_repeats"])
MODEL_CLASS = OPTIONS["model"]
OUTPUT_FILE = OPTIONS["output_file"]
DISTRIBUTE_CONDITIONS = bool(OPTIONS["distribute_conditions"])
#%% Determine the model to fit and parameter bounds
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

NUM_PARAMS = len(combined_model.PARAMETERS)
LB, UB = combined_model.get_parameter_bounds()

l10LB = np.log10(LB)
l10UB = np.log10(UB)

#%% Determine starting point
dv0 = np.zeros((NUM_PARAMS,))
if OPTIONS["starting_point"]:
    filename = OPTIONS["starting_point"]
    with np.load(filename) as f:
        dv0[:] = f["par"]
else:
    dv0 = np.random.uniform(low=LB, high=UB)
#%%
DATA = {}
for species in FIT_SPECIES:
    DATA[species] = {}
    for condi in FIT_CONDITIONS:
        DATA[species][condi] = [
            SmFishSnapshot(a) for a in load_single_species_data(condi, species)
        ]


def solve_single_gene_cme(
    comm: mpi.Comm,
    model: Union[
        ThreeStateSingleGeneA,
        ThreeStateSingleGeneB,
        ThreeStateSingleGeneC,
        ThreeStateSingleGeneD,
    ],
    tnodes: np.ndarray,
) -> [DiscreteDistribution]:
    propensity_t = model.propensity_t
    propensity_x = model.propensity_x
    cme_solver = FspSolverMultiSinks(comm)
    cme_solver.SetModel(model.STOICH_MAT, propensity_t, propensity_x)
    cme_solver.SetFspShape(constr_fun=None, constr_bound=model.INIT_BOUNDS)
    cme_solver.SetInitialDist(model.X0, model.P0)
    cme_solver.SetLBMethod("GRAPH")
    cme_solver.SetOdeSolver("PETSC")
    cme_solver.SetOdeTolerances(1.0e-4, 1.0e-8)
    cme_solver.SetVerbosity(0)
    cme_solver.SetUp()
    solutions = cme_solver.SolveTspan(tnodes + model.T0, 1.0e-7)
    cme_solver.ClearState()
    return solutions


def single_species_condition_loglike(
    comm: mpi.Comm,
    theta: Union[np.ndarray, dict],
    model: Any,
    species: str,
    condition: str,
) -> float:
    try:
        solutions = solve_single_gene_cme(comm, model(theta), model.T_MEASUREMENTS)
    except RuntimeError:
        return 1.0e8

    if solutions is None:
        return 1.0e8

    ll = 0.0
    for time_idx in range(0, len(solutions)):
        ll = ll + DATA[species][condition][time_idx].LogLikelihood(
            solutions[time_idx], np.array([RNA_IDX])
        )
    return ll


BEST_LL = -1.0e8


def compute_objective(comm: mpi.Comm, dv: np.ndarray) -> float:
    ll = 0.0
    if not DISTRIBUTE_CONDITIONS:
        for my_species in FIT_SPECIES:
            for condition in FIT_CONDITIONS:
                single_gene_model = getattr(combined_model, f"{my_species.upper()}_SRN")
                theta_single_gene = combined_model.convert_to_single_gene_parameters(10.0**dv, my_species, condition)
                ll += single_species_condition_loglike(
                    comm, theta_single_gene, single_gene_model, my_species, condition
                )
    else:
        color = comm.Get_rank() % 6
        subrank = comm.Get_rank() // 6

        smaller_comm = comm.Split(color, subrank)

        my_species = FIT_SPECIES[color // 3]
        condition = FIT_CONDITIONS[color % 3]

        single_gene_model = getattr(combined_model, f"{my_species.upper()}_SRN")
        theta_single_gene = combined_model.convert_to_single_gene_parameters(10.0 ** dv, my_species, condition)
        ll_local = single_species_condition_loglike(
            smaller_comm, theta_single_gene, single_gene_model, my_species, condition
        )

        if smaller_comm.Get_rank() != 0:
            ll_local = 0.0
        ll = comm.allreduce(ll_local)

    global BEST_LL
    if ll >= BEST_LL:
        BEST_LL = ll
        if WORLD.Get_rank() == 0:
            np.savez(
                f"{OUTPUT_FILE}", par=dv, fitness=-1.0 * ll,
            )
    return -1.0 * ll


#%%
if __name__ == "__main__":
    WORLD = mpi.COMM_WORLD
    compass_search = pygmo.algorithm(
        pygmo.compass_search(max_fevals=2500, start_range=1.0, stop_range=1.0e-4)
    )
    compass_search.set_verbosity(1)
    annealing = pygmo.algorithm(pygmo.simulated_annealing())
    annealing.set_verbosity(1)

    for i_opt in range(0, OPT_REPEATS):
        for algorithm in [annealing, compass_search]:
            dv_opt, f_opt = Minimize(
                WORLD, (l10LB, l10UB), dv0, compute_objective, algorithm
            )

            if WORLD.Get_rank() == 0:
                np.savez(
                    OUTPUT_FILE, par=dv_opt, fitness=f_opt,
                )
            dv0 = dv_opt
