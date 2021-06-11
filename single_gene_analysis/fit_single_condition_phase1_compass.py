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
    CircularThreeStateSingleGeneA,
    CircularThreeStateSingleGeneB
)
import mpi4py.MPI as mpi
import pygmo
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.smfish.snapshot import SmFishSnapshot
from modules.data_io.data_io import load_single_species_data
from fit_settings import *
import matplotlib.pyplot as plt

#%% Code for worker's mode
EVOLVE_POPULATION = 102
QUIT = -1
#%%
OPTIONS = {
    "fit_problem": "3SA_il1b_NoInhibitors",
    "output_file": "compass_opt.npz",
    "monitor_dir": ".",
    "initial_pop": None,
    "num_epochs": 100,
    "max_fevals": 100,
    "start_range": 0.1,
    "stop_range": 1e-5,
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
NUM_EPOCHS = int(OPTIONS["num_epochs"])
OUTPUT_FILE = OPTIONS["output_file"]
INIT_POP_F = OPTIONS["initial_pop"]
MONITOR_DIR = OPTIONS["monitor_dir"]
#%%
if INIT_POP_F is None:
    raise RuntimeError("Must give path to initial population file.")

with np.load(INIT_POP_F) as file:
    init_xs = file["xs"]
    init_fs = file["fs"]
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
elif MODEL_CLASS == "C3SA":
    MODEL = CircularThreeStateSingleGeneA
    RNA_IDX = 3
elif MODEL_CLASS == "C3SB":
    MODEL = CircularThreeStateSingleGeneB
    RNA_IDX = 3
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
    solutions = cme_solver.SolveTspan(model.T_MEASUREMENTS + model.T0, 1.0e-6)
    return solutions


SMFISHDATA = [SmFishSnapshot(a) for a in load_single_species_data(CONDITION, SPECIES)]


def compute_objective(dv) -> float:
    log10theta = dv

    try:
        solutions = solve_model(mpi.COMM_SELF, log10theta)
    except:
        return 1.0e8

    if solutions is None:
        return 1.0e8

    ll = 0.0
    for i in range(0, len(solutions)):
        ll = ll + SMFISHDATA[i].LogLikelihood(solutions[i], np.array([RNA_IDX]))

    return -1.0 * ll


#%% Interface the CME likelihood to pyGMO
class OptProblem:
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds

    def get_bounds(self):
        return self.bounds[0], self.bounds[1]

    def fitness(self, dv):
        fitness = compute_objective(dv)
        return [fitness]


def batch_evolve(dvs: np.ndarray, start_ranges: float) -> (np.ndarray, np.ndarray, np.ndarray):
    dvs = dvs.reshape((-1, MODEL.NUM_PARAMETERS))

    num_evals = dvs.shape[0]

    dvs_new = np.zeros((num_evals, MODEL.NUM_PARAMETERS))
    vals = np.zeros((num_evals,))
    end_ranges = np.zeros((num_evals,))

    report_buf = np.zeros((1,), dtype=int)
    command_buf = np.array([EVOLVE_POPULATION], dtype=int)

    input_buf = np.zeros((MODEL.NUM_PARAMETERS + 1,), dtype=float)
    eval_buf = np.zeros((MODEL.NUM_PARAMETERS + 2,), dtype=float)

    report_requests = []
    command_requests = []
    input_send_requests = []
    eval_recv_requests = []
    for i in range(NUM_WORKERS - 1):
        report_requests.append(WORLD.Recv_init([report_buf, 1, mpi.INT], i + 1))
        command_requests.append(WORLD.Send_init([command_buf, 1, mpi.INT], i + 1))
        input_send_requests.append(
            WORLD.Send_init([input_buf, MODEL.NUM_PARAMETERS + 1, mpi.DOUBLE], i + 1)
        )
        eval_recv_requests.append(
            WORLD.Recv_init([eval_buf, MODEL.NUM_PARAMETERS + 2, mpi.DOUBLE], i + 1)
        )

    for i in range(NUM_WORKERS - 1):
        report_requests[i].Start()

    recv_address = [0] * NUM_WORKERS
    k = 0
    stat = mpi.Status()
    while k < num_evals:
        # Check for available CPUs
        mpi.Prequest.Waitany(report_requests, stat)
        source_rank = stat.Get_source()
        if source_rank > 0:
            recv_address[source_rank - 1] = k
            input_buf[:-1] = dvs[k, :]
            input_buf[-1] = start_ranges[k]
            command_requests[source_rank - 1].Start()
            command_requests[source_rank - 1].Wait()
            input_send_requests[source_rank - 1].Start()
            input_send_requests[source_rank - 1].Wait()
            eval_recv_requests[source_rank - 1].Start()
            k += 1
        else:
            # Check for returning results
            mpi.Prequest.Waitany(eval_recv_requests, stat)
            source_rank = stat.Get_source()
            if source_rank > 0:
                dvs_new[recv_address[source_rank - 1], :] = eval_buf[0:-2]
                vals[recv_address[source_rank - 1]] = eval_buf[-2]
                end_ranges[recv_address[source_rank - 1]] = eval_buf[-1]
                report_requests[source_rank - 1].Start()
    while True:
        mpi.Prequest.Waitany(eval_recv_requests, stat)
        source_rank = stat.Get_source()
        if source_rank > 0:
            dvs_new[recv_address[source_rank - 1], :] = eval_buf[0:-2]
            vals[recv_address[source_rank - 1]] = eval_buf[-2]
            end_ranges[recv_address[source_rank-1]] = eval_buf[-1]
        else:
            break
    return dvs_new, vals, end_ranges


#%%
if __name__ == "__main__":
    LB, UB = [np.log10(b) for b in MODEL.get_parameter_bounds()]
    PROBLEM = pygmo.problem(OptProblem(MODEL.NUM_PARAMETERS, [LB, UB]))
    if MY_RANK == 0:
        xs = init_xs.copy()
        ranges = float(OPTIONS["start_range"])*np.ones((xs.shape[0],))
        print(ranges)
        for i in range(NUM_EPOCHS):
            xs, fs, ranges = batch_evolve(xs, ranges)
            np.savez(OUTPUT_FILE, xs=xs, fs=fs)

            fig, ax = plt.subplots(1, 1)
            ax.hist(fs)
            fig.savefig(f"{MONITOR_DIR}/{FIT_PROBLEM}_epoch_{i}.png")
            plt.close(fig)

        # Tell the worker processes to call it a day
        command_buf = np.array([QUIT], dtype=np.int32)
        for i in range(1, NUM_WORKERS):
            WORLD.Send([command_buf, 1, mpi.INT], i)
    else:
        report_buf = np.zeros((1,), dtype=np.int32)
        root_buf = np.zeros((1,), dtype=np.int32)
        while True:
            # Report to Master
            WORLD.Send([report_buf, 1, mpi.INT], 0)
            # Wait for command from Master
            WORLD.Recv([root_buf, 1, mpi.INT], 0)
            if root_buf[0] == EVOLVE_POPULATION:
                # In Evolution mode - run optimization algorithm on the population sent from Master
                dv_buf = np.zeros((MODEL.NUM_PARAMETERS+1,))
                WORLD.Recv([dv_buf, MODEL.NUM_PARAMETERS+1, mpi.DOUBLE], source=0)

                compass_opts = {
                    "max_fevals": int(OPTIONS["max_fevals"]),
                    "start_range": dv_buf[-1],
                    "stop_range": float(OPTIONS["stop_range"]),
                }
                if compass_opts["start_range"] <= compass_opts["stop_range"]:
                    compass_opts["start_range"]*= 2.0
                compass = pygmo.algorithm(pygmo.compass_search(**compass_opts))
                compass.set_verbosity(1)

                pop = pygmo.population(PROBLEM)
                pop.push_back(dv_buf[0:-1])
                pop = compass.evolve(pop)
                x = pop.champion_x
                f = pop.champion_f
                uda = compass.extract(pygmo.compass_search)
                stop_range = compass.extract(pygmo.compass_search).get_log()[-1][-1]
                return_buf = np.hstack((x, f, np.array(stop_range)))
                WORLD.Send([return_buf, MODEL.NUM_PARAMETERS + 2, mpi.DOUBLE], dest=0)
            else:
                break
