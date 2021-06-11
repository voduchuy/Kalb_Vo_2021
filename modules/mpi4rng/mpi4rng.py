import mpi4py.MPI as mpi
import numpy as np
from numpy.random import default_rng, SeedSequence

class mpiGenerator:
    def __init__(self, comm: mpi.Comm, seed: int = None):
        self.comm = comm.Dup()

        if self.comm.Get_rank() == 0:
            # Initialize random generator
            seq = SeedSequence(seed)
            ss = [default_rng(s) for s in seq.spawn(self.comm.Get_size())]
        else:
            ss = None
        self.rng_state = comm.scatter(ss, root=0)

