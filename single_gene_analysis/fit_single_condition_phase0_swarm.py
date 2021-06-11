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

#%%
OPTIONS = {
    "fit_problem": "3SA_il1b_NoInhibitors",
    "output_file": "de_opt.npz",
    "starting_population": None,
    "num_generations": 100,
    "num_epochs": 10,
    "population_size": 100,
    "monitor_dir": ".",
    "neighb_type": 2,
    "variant": 6,
    "eta1": 1,
    "eta2": 3,
    "max_vel": 0.5,
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
NUM_GENERATIONS = int(OPTIONS["num_generations"])
NUM_EPOCHS = int(OPTIONS["num_epochs"])
POPULATION_SIZE = int(OPTIONS["population_size"])
MONITOR_DIR = OPTIONS["monitor_dir"]
SWARM_NEIGHB_TYPE = int(OPTIONS["neighb_type"])
VARIANT = int(OPTIONS["variant"])
ETA1 = float(OPTIONS["eta1"])
ETA2 = float(OPTIONS["eta2"])
MAX_VEL = float(OPTIONS["max_vel"])
#%%
WORLD = mpi.COMM_WORLD
MY_RANK = WORLD.Get_rank()
NUM_WORKERS = WORLD.Get_size() - 1

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
if OPTIONS["starting_population"] is not None:
    with np.load(OPTIONS["starting_population"], allow_pickle=True) as f:
        pop0 = [f["xs"], f["fs"]]
        POPULATION_SIZE = f["xs"].shape[0]
else:
    pop0 = None

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
    except RuntimeError:
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
        self.work_signal = np.array([0], dtype=int)

    def get_bounds(self):
        return self.bounds[0], self.bounds[1]

    def fitness(self, dv):
        self.work_signal[0] = 1
        fitness = compute_objective(mpi.COMM_SELF, dv)
        return [fitness]

    def batch_fitness(self, dvs):
        dvs = dvs.reshape((-1, MODEL.NUM_PARAMETERS))
        num_evals = dvs.shape[0]
        vals = np.zeros((num_evals,))
        report_buf = np.zeros((1,), dtype=int)
        command_buf = np.zeros((1,), dtype=int)
        dv_buf = np.zeros((MODEL.NUM_PARAMETERS,), dtype=float)
        eval_buf = np.zeros((1,), dtype=float)

        report_requests = []
        command_requests = []
        dv_send_requests = []
        eval_recv_requests = []
        for i in range(NUM_WORKERS):
            report_requests.append(WORLD.Recv_init([report_buf, 1, mpi.INT], i + 1))
            command_requests.append(WORLD.Send_init([command_buf, 1, mpi.INT], i + 1))
            dv_send_requests.append(
                WORLD.Send_init([dv_buf, MODEL.NUM_PARAMETERS, mpi.DOUBLE], i + 1)
            )
            eval_recv_requests.append(WORLD.Recv_init([eval_buf, 1, mpi.DOUBLE], i + 1))

        for i in range(NUM_WORKERS):
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
                dv_buf[:] = dvs[k, :]
                command_requests[source_rank - 1].Start()
                command_requests[source_rank - 1].Wait()
                dv_send_requests[source_rank - 1].Start()
                dv_send_requests[source_rank - 1].Wait()
                eval_recv_requests[source_rank - 1].Start()
                k += 1
            else:
                # Check for returning results
                mpi.Prequest.Waitany(eval_recv_requests, stat)
                source_rank = stat.Get_source()
                if source_rank > 0:
                    vals[recv_address[source_rank - 1]] = eval_buf[0]
                    report_requests[source_rank - 1].Start()

        while True:
            mpi.Prequest.Waitany(eval_recv_requests, stat)
            source_rank = stat.Get_source()
            if source_rank > 0:
                vals[recv_address[source_rank - 1]] = eval_buf[0]
            else:
                break

        return vals


#%%
if __name__ == "__main__":
    if MY_RANK == 0:
        LB, UB = [np.log10(b) for b in MODEL.get_parameter_bounds()]

        problem = pygmo.problem(OptProblem(MODEL.NUM_PARAMETERS, [LB, UB]))

        if pop0 is not None:
            pop = pygmo.population(problem)
            for i in range(POPULATION_SIZE):
                pop.push_back(pop0[0][i, :], pop0[1][i])
        else:
            pop = pygmo.population(problem, size=POPULATION_SIZE, b=pygmo.bfe())

        algo = pygmo.pso_gen(
            gen=NUM_GENERATIONS,
            eta1=ETA1,
            eta2=ETA2,
            memory=True,
            variant=VARIANT,
            neighb_type=SWARM_NEIGHB_TYPE,
            max_vel=MAX_VEL,
        )
        algo.set_bfe(pygmo.bfe())
        algo = pygmo.algorithm(algo)
        algo.set_verbosity(1)

        import matplotlib.pyplot as plt

        for epoch in range(NUM_EPOCHS):
            print(f"EPOCH {epoch}:\n")
            pop = algo.evolve(pop)
            xs = pop.get_x()
            fs = pop.get_f()
            xbest = pop.champion_x
            fbest = pop.champion_f
            iworst = pop.worst_idx()
            xs[iworst, :] = xbest[:]
            fs[iworst, :] = fbest[:]
            np.savez(OUTPUT_FILE, xs=xs, fs=fs, allow_pickle=True)

            fig, ax = plt.subplots(1, 1)
            ax.hist(pop.get_f())
            fig.savefig(f"{MONITOR_DIR}/{FIT_PROBLEM}_epoch_{epoch}.png")
            plt.close(fig)

        # Tell the worker processes to call it a day
        command_buf = np.array([-1], dtype=np.int32)
        for i in range(NUM_WORKERS):
            WORLD.Isend([command_buf, 1, mpi.INT], i + 1)
    else:
        report_buf = np.zeros((1,), dtype=np.int32)
        root_buf = np.zeros((1,), dtype=np.int32)

        input_buf = np.zeros((MODEL.NUM_PARAMETERS,), dtype=float)
        return_buf = np.zeros((1,), dtype=float)

        while True:
            request = WORLD.Isend([report_buf, 1, mpi.INT], 0)
            request.Wait()
            request = WORLD.Irecv([root_buf, 1, mpi.INT], 0)
            request.Wait()

            if root_buf[0] == -1:
                break
            else:
                WORLD.Recv([input_buf, len(input_buf), mpi.DOUBLE], 0)
                return_buf[0] = compute_objective(mpi.COMM_SELF, input_buf)
                request = WORLD.Isend([return_buf, 1, mpi.DOUBLE], 0)
                request.Wait()
