import sys
sys.path.append("..")
sys.path.append("../modules/")
from models import (
    TwoStateSingleGeneA,
    TwoStateSingleGeneB,
    ThreeStateSingleGeneA,
    ThreeStateSingleGeneB,
    ThreeStateSingleGeneC,
    ThreeStateSingleGeneD,
)
import mpi4py.MPI as mpi
import pygmo
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.smfish.snapshot import SmFishSnapshot
from modules.data_io.data_io import load_single_species_data
from modules.mpi4pygmo import Minimize
from fit_settings import *

WORLD = mpi.COMM_WORLD
#%%
OPTIONS = {
    "fit_problem": "3SA_il1b_NoInhibitors",
    "output_file": "fit.npz",
    "starting_point": None,
    "search_repeats": 1,
}
#%%
ARGV = sys.argv

for i in range(1, len(ARGV)):
    key, value = ARGV[i].split("=")
    if key in OPTIONS:
        OPTIONS[key] = value
    else:
        print(f"WARNING: Unknown option {key} \n")

FIT_PROBLEM = OPTIONS["fit_problem"]
OUTPUT_FILE = OPTIONS["output_file"]
OPT_REPEATS = int(OPTIONS["search_repeats"])
#%%
WORLD = mpi.COMM_WORLD
MY_RANK = WORLD.Get_rank()
NUM_WORKERS = WORLD.Get_size()

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

#%%
def generate_initial_theta():
    if OPTIONS["starting_point"]:
        with np.load(OPTIONS["starting_point"]) as file:
            log10theta = file["par"]
    else:
        LB, UB = MODEL.get_parameter_bounds()
        log10theta = np.random.uniform(np.log10(LB), np.log10(UB))

    return log10theta


LOG10THETA0 = generate_initial_theta()
#%%


def solve_model(comm: mpi.Comm, log10theta: np.ndarray):
    model = MODEL(10.0 ** log10theta)
    propensity_t = model.propensity_t
    propensity_x = model.propensity_x
    cme_solver = FspSolverMultiSinks(comm)
    cme_solver.SetModel(model.STOICH_MAT, propensity_t, propensity_x)
    cme_solver.SetFspShape(constr_fun=None, constr_bound=model.INIT_BOUNDS)
    cme_solver.SetInitialDist(model.X0, model.P0)
    cme_solver.SetOdeSolver("PETSC")
    solutions = cme_solver.SolveTspan(model.T_MEASUREMENTS + model.T0, 1.0e-4)
    return solutions


SMFISHDATA = [SmFishSnapshot(a) for a in load_single_species_data(CONDITION, SPECIES)]


def compute_objective(comm: mpi.Comm, dv) -> float:
    log10theta = dv

    try:
        solutions = solve_model(comm, log10theta)
    except:
        return 1.0e8

    if solutions is None:
        return 1.0e8

    ll = 0.0
    for i in range(0, len(solutions)):
        ll = ll + SMFISHDATA[i].LogLikelihood(solutions[i], np.array([RNA_IDX]))
    return -1.0 * ll


#%%
if __name__ == "__main__":
    dv0 = LOG10THETA0
    LB, UB = [np.log10(b) for b in MODEL.get_parameter_bounds()]

    compass_search = pygmo.algorithm(
        pygmo.compass_search(max_fevals=10000, start_range=1.0, stop_range=1.0e-4)
    )
    compass_search.set_verbosity(1)

    annealing = pygmo.algorithm(pygmo.simulated_annealing())
    annealing.set_verbosity(1)

    for i_opt in range(0, OPT_REPEATS):
        for algorithm in [annealing, compass_search]:
            # Run with compass search
            dv_opt, f_opt = Minimize(WORLD, (LB, UB), dv0, compute_objective, algorithm)

            if WORLD.Get_rank() == 0:
                log10theta = dv_opt
                np.savez(
                    OUTPUT_FILE, par=log10theta, dv=dv_opt, fitness=f_opt,
                )
            dv0 = dv_opt
