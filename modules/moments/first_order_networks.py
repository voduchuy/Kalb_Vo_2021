# Implement moment equation solving for first-order linear models
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
import numpy as np
from typing import Callable, Any


def find_mean_var(
    stoichMatrix: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    timeSignal: Callable[[float, np.ndarray], Any],
    timeEval: np.ndarray,
    initMean: np.ndarray,
    initCovariance: np.ndarray,
    odeMethod: str = "BDF",
    relTol: float = 1.0e-4,
    absTol: float = 1.0e-8,
) -> [np.ndarray, np.ndarray]:
    numSpecies = stoichMatrix.shape[1]

    if b is None:
        b = np.zeros((stoichMatrix.shape[0],))

    def odeRhs(t, y):
        dydt = np.zeros(y.shape)
        tCoeffVec = np.zeros((stoichMatrix.shape[0], 1))
        timeSignal(t, tCoeffVec)
        W1 = tCoeffVec * W

        mu = y[0:numSpecies]
        sigma2 = y[numSpecies:]
        sigma2 = sigma2.reshape((numSpecies, numSpecies), order="F")

        M1 = stoichMatrix.T.dot(W1 @ sigma2)
        M1 += M1.T

        M1 += (stoichMatrix.T) @ (np.diag(W1.dot(mu) + b)) @ (stoichMatrix)

        dydt[0:numSpecies] = stoichMatrix.T.dot(W1.dot(mu) + b)
        dydt[numSpecies:] = M1.reshape((-1,))

        return dydt

    y0 = np.zeros((numSpecies + numSpecies ** 2))
    y0[0:numSpecies] = initMean
    y0[numSpecies:] = initCovariance.reshape((-1,))

    problem = Explicit_Problem(odeRhs, y0, 0.0)
    solver = CVode(problem)
    solver.verbosity = 100

    tout, Y = solver.simulate(timeEval[-1], 0, timeEval)
    Y = Y[1:, :]

    meanTrajectory = Y[:, 0:numSpecies]
    covarianceTrajectory = Y[:, numSpecies:].reshape(
        (-1, numSpecies, numSpecies), order="C"
    )
    return meanTrajectory, covarianceTrajectory
