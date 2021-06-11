import sys
import numpy as np
import mpi4py.MPI as mpi
import pygmo


#%%
OPTIONS = {
        'output_file': 'de_opt.npz',
        'num_generations': 100,
        'num_epochs': 10,
        'population_size': 100,
        'monitor_dir': '.'
}
#%%
ARGV = sys.argv

for i in range(1, len(ARGV)):
    key, value = ARGV[i].split('=')
    if key in OPTIONS:
        OPTIONS[key] = value
    else:
        print(f"WARNING: Unknown option {key} \n")

OUTPUT_FILE = OPTIONS['output_file']
NUM_GENERATIONS = int(OPTIONS['num_generations'])
NUM_EPOCHS = int(OPTIONS['num_epochs'])
POPULATION_SIZE = int(OPTIONS['population_size'])
MONITOR_DIR = OPTIONS['monitor_dir']
#%%
WORLD = mpi.COMM_WORLD
MY_RANK = WORLD.Get_rank()
NUM_WORKERS = WORLD.Get_size()-1

#%%
def compute_objective(dv: np.ndarray) -> float:
    return np.sum(dv*dv)[0]
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
        fitness = compute_objective(dv)
        return [fitness]

    def batch_fitness(self, dvs):
        num_evals = len(dvs)//8
        vals = np.zeros((num_evals,))
        report_buf = np.zeros((1,), dtype=int)
        command_buf = np.zeros((1,), dtype=int)
        dv_buf = np.zeros((2,), dtype=float)
        eval_buf = np.zeros((1,), dtype=float)

        report_requests = []
        command_requests = []
        dv_send_requests = []
        eval_recv_requests = []
        for i in range(NUM_WORKERS):
            report_requests.append(WORLD.Recv_init([report_buf, 1, mpi.INT], i + 1))
            command_requests.append(WORLD.Send_init([command_buf, 1, mpi.INT], i + 1))
            dv_send_requests.append(WORLD.Send_init([dv_buf, 2, mpi.DOUBLE], i + 1))
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
                dv_buf[:] = dvs[2 * k:2 * k + 2]
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
        LB, UB = [np.array([-1]*8), np.array([1]*8)]

        problem = pygmo.problem(OptProblem(8, [LB, UB]))


        pop = pygmo.population(problem, size=POPULATION_SIZE, b=pygmo.bfe())

        algo = pygmo.pso_gen(gen=NUM_GENERATIONS, memory=True, neighb_type=4)
        algo.set_bfe(pygmo.bfe())
        algo = pygmo.algorithm(algo)
        algo.set_verbosity(1)

        import matplotlib.pyplot as plt
        for epoch in range(NUM_EPOCHS):
            print(f'EPOCH {epoch}:\n')
            pop = algo.evolve(pop)
            xs = pop.get_x()
            fs = pop.get_f()
            xbest = pop.champion_x
            fbest = pop.champion_f
            iworst = pop.worst_idx()
            xs[iworst,:] = xbest[:]
            fs[iworst,:] = fbest[:]
            np.savez(OUTPUT_FILE, xs=xs, fs=fs, allow_pickle=True)

            fig, ax = plt.subplots(1,1)
            ax.hist(pop.get_f())
            fig.savefig(f"{MONITOR_DIR}/epoch_{epoch}.png")

        # Tell the worker processes to call it a day
        command_buf = np.array([-1], dtype=np.int32)
        for i in range(NUM_WORKERS):
            WORLD.Isend([command_buf, 1, mpi.INT], i+1)
    else:
        report_buf = np.zeros((1,), dtype=np.int32)
        root_buf = np.zeros((1,), dtype=np.int32)

        input_buf = np.zeros((8,), dtype=float)
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
                return_buf[0] = compute_objective(input_buf)
                request = WORLD.Isend([return_buf, 1, mpi.DOUBLE], 0)
                request.Wait()

