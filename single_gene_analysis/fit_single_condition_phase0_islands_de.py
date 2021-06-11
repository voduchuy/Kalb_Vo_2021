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
from fit_settings import *

#%% Code for worker's mode
BATCH_EVALUATION = 101
EVOLVE_POPULATION = 102
QUIT = -1
#%%
OPTIONS = {
    "fit_problem": "3SA_il1b_NoInhibitors",
    "output_file": "de_opt.npz",
    "num_generations": 100,
    "num_epochs": 10,
    "population_size": 100,
    "monitor_dir": ".",
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

    def batch_fitness(self, dvs):
        dvs = dvs.reshape((-1, MODEL.NUM_PARAMETERS))
        num_evals = dvs.shape[0]
        vals = np.zeros((num_evals,))
        report_buf = np.zeros((1,), dtype=int)
        command_buf = np.array([BATCH_EVALUATION], dtype=int)
        dv_buf = np.zeros((MODEL.NUM_PARAMETERS,), dtype=float)
        eval_buf = np.zeros((1,), dtype=float)

        report_requests = []
        command_requests = []
        dv_send_requests = []
        eval_recv_requests = []
        for i in range(NUM_WORKERS - 1):
            report_requests.append(WORLD.Recv_init([report_buf, 1, mpi.INT], i + 1))
            command_requests.append(WORLD.Send_init([command_buf, 1, mpi.INT], i + 1))
            dv_send_requests.append(
                WORLD.Send_init([dv_buf, MODEL.NUM_PARAMETERS, mpi.DOUBLE], i + 1)
            )
            eval_recv_requests.append(WORLD.Recv_init([eval_buf, 1, mpi.DOUBLE], i + 1))

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


#%% MPI Island type
class MPIIsland:
    def __init__(self, rank: int):
        self.rank = rank

    def run_evolve(self, algo, pop):
        command_buf = np.array([EVOLVE_POPULATION], dtype=np.int32)
        report_buf = np.zeros((1,), dtype=np.int32)
        if self.rank == 0:
            pop = algo.evolve(pop)
        else:
            WORLD.Recv([report_buf, 1, mpi.INT], self.rank)
            WORLD.Send([command_buf, 1, mpi.INT], self.rank)
            xs = pop.get_x()
            fs = pop.get_f()
            WORLD.send(xs, dest=self.rank, tag=self.rank)
            WORLD.send(fs, dest=self.rank, tag=self.rank + NUM_WORKERS)
            new_xs = WORLD.recv(source=self.rank, tag=self.rank)
            new_fs = WORLD.recv(source=self.rank, tag=self.rank + NUM_WORKERS)
            for i in range(xs.shape[0]):
                pop.set_xf(i, new_xs[i, :], new_fs[i])
        return algo, pop


#%%
if __name__ == "__main__":
    ALGO = pygmo.algorithm(pygmo.de(gen=NUM_GENERATIONS))
    LB, UB = [np.log10(b) for b in MODEL.get_parameter_bounds()]
    PROBLEM = pygmo.problem(OptProblem(MODEL.NUM_PARAMETERS, [LB, UB]))
    if MY_RANK == 0:

        island_opts = {
            "algo": ALGO,
            "prob": PROBLEM,
            "size": POPULATION_SIZE,
            "b": pygmo.bfe(),
        }
        islands = [
            pygmo.island(udi=MPIIsland(rank), **island_opts)
            for rank in range(NUM_WORKERS)
        ]

        archi = pygmo.archipelago()
        for island in islands:
            archi.push_back(island)
        archi.set_topology(pygmo.fully_connected(n=len(islands)))

        import matplotlib.pyplot as plt

        for epoch in range(NUM_EPOCHS):
            archi.evolve()
            archi.wait_check()
            xs = np.array(archi.get_champions_x())
            fs = np.array(archi.get_champions_f())
            np.savez(OUTPUT_FILE, xs=xs, fs=fs, allow_pickle=True)

            print(f"{epoch}  best f = {np.min(fs)}")

            fig, ax = plt.subplots(1, 1)
            ax.hist(fs)
            fig.savefig(f"{MONITOR_DIR}/{FIT_PROBLEM}_epoch_{epoch}.png")
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
                xs = WORLD.recv(source=0, tag=MY_RANK)
                fs = WORLD.recv(source=0, tag=MY_RANK + NUM_WORKERS)
                pop = pygmo.population(PROBLEM)
                for i in range(0, xs.shape[0]):
                    pop.push_back(xs[i, :], fs[i])
                pop = ALGO.evolve(pop)
                xs = pop.get_x()
                fs = pop.get_f()
                WORLD.send(xs, dest=0, tag=MY_RANK)
                WORLD.send(fs, dest=0, tag=MY_RANK + NUM_WORKERS)
            elif root_buf[0] == BATCH_EVALUATION:
                # In Batch evaluation mode - evaluate fitness of individuals sent from Master
                input_buf = np.zeros((MODEL.NUM_PARAMETERS,), dtype=float)
                return_buf = np.zeros((1,), dtype=float)
                WORLD.Recv([input_buf, len(input_buf), mpi.DOUBLE], 0)
                return_buf[0] = compute_objective(input_buf)
                request = WORLD.Send([return_buf, 1, mpi.DOUBLE], 0)
            else:
                break
