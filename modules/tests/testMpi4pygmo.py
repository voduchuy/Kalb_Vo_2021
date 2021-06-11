import sys

sys.path.append("../..")
import mpi4py.MPI as mpi
import modules.mpi4pygmo as mpi4pygmo
import numpy as np
import pygmo
import matplotlib.pyplot as plt
from modules.mpi4rng import mpiGenerator

N_PER_PROC = 100000

mpi_rng = mpiGenerator(mpi.COMM_WORLD, 0)
rng = mpi_rng.rng_state


def objective(comm: mpi.Comm, dv: np.ndarray) -> float:
    mu = dv[0]
    sigma = dv[1]

    x_samples = rng.normal(loc=mu, scale=sigma, size=(N_PER_PROC,))

    partial_sum = (1 / (N_PER_PROC * mpi.COMM_WORLD.Get_size())) * np.sum(
        -((x_samples - mu) ** 2.0) / (2 * sigma * sigma)
        + x_samples ** 2.0 / 2
        - np.log(sigma)
    )

    kl_est = comm.allreduce(partial_sum, op=mpi.SUM)

    return kl_est


comm = mpi.COMM_WORLD
bounds = (np.array([-5.0, 0.001]), np.array([5, 5]))
initGuess = np.array([0.5, 0.5])
obj = objective
algo = pygmo.algorithm(
    pygmo.compass_search(max_fevals=2000, start_range=1.0, stop_range=1.0e-6)
)
algo.set_verbosity(1)
thetaOpt, fitness = mpi4pygmo.Minimize(comm, bounds, initGuess, obj, algo)

if comm.Get_rank() == 0:
    print(thetaOpt)
