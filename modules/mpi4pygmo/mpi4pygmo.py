import mpi4py.MPI as mpi
import pygmo
import numpy as np
from typing import Callable

def Minimize(comm: mpi.Comm,
             bounds: ([np.ndarray], [np.ndarray]),
             init_guess: np.ndarray,
             objective: Callable[[mpi.Comm, np.ndarray], float],
             algo: pygmo.algorithm) -> (np.ndarray, float):
    class Problem:
        def __init__(self):
            self.dim = len(init_guess)
            self.work_signal = np.array([0], dtype=int)

        def get_bounds(self):
            return bounds[0], bounds[1]

        def fitness(self, dv):
            self.work_signal[0] = 1
            # Send the inputs to worker processes
            comm.Bcast(self.work_signal, root=0)
            comm.Bcast(dv, root=0)
            fitness = objective(comm, dv)
            return [fitness]

        def gradient(self, x):
            return pygmo.estimate_gradient(lambda x: self.fitness(x), x)  # we here use the low precision gradient

    rank = comm.Get_rank()
    work_signal = np.array([1], dtype=int)
    problem = pygmo.problem(Problem())
    pop = pygmo.population(problem)
    dim_theta = len(init_guess)
    if rank == 0:
        pop.push_back(init_guess)
        pop = algo.evolve(pop)
        # Tell the worker processes to call it a day
        work_signal[0] = -1
        comm.Bcast(work_signal, root=0)
        # Broadcast optimal solution to all processes
        return_val = comm.bcast(pop.champion_x)
        return_fitness = comm.bcast(pop.champion_f)
    else:
        dv = np.empty(dim_theta, dtype=float)
        y = np.array([0], dtype=float)
        wait_for_task = True
        while wait_for_task:
            comm.Bcast(work_signal, root=0)
            if work_signal[0] == 1:
                comm.Bcast(dv, root=0)
                y[0] = objective(comm, dv)
            elif work_signal[0] == -1:
                wait_for_task = False
        return_val = comm.bcast(None)
        return_fitness = comm.bcast(None)
    return return_val, return_fitness
