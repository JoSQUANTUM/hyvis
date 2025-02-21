"""This module collects some examples of loss functions,
as well as a simple gradient descent function."""

from typing import Callable

import numpy as np
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.providers.basic_provider import BasicProvider

from hyvis.dr_tools import AffineSubspace, numeric_gradient


def gradient_descent_training(
    func: Callable[[np.ndarray], float],
    subspace: AffineSubspace,
    rate: float,
    steps: int,
    epsilon: float,
):
    """This function caclulates a training trajectory using gradient descent.

    Input:
    - func: any function mapping an array of real numbers to a real number
    - subspace: an affine linear subspace of the parameter space,
        its center is the initial state of the trainging and
        its directions are the allowed directions in the training
    - rate: the learning rate, ie the factor put on the gradient in each step
    - steps: the number of steps taken, including the initial step
    - epsilon: the offset used in the approximation, should either be small
        or chosen according to some parameter shift rule

    """

    dim = subspace.center.shape[1]

    traj = np.ndarray(shape=(steps, dim))

    # taking the negative of the loss landscape to get actual descent
    def desfunc(arg):
        return -func(arg)

    for step_id in range(steps):
        traj[step_id, :] = subspace.center
        subspace.center = subspace.center + rate * numeric_gradient(
            func=desfunc, subspace=subspace, epsilon=epsilon, relative=True
        )

    return traj


def relative_entropy_univariate_gaussians(
    par: np.ndarray,
) -> float:
    """Calculates the KL divergence of two univariate Gaussians,
    given their mean and variance."""

    par = par.flatten()

    if not par.shape == (4,) or not par[1] > 0 or not par[3] > 0:
        raise AttributeError(
            "Parameters must be given as np.array([mu1, sigma1^2, mu2, sigma2^2])."
        )

    return (
        (par[1] / par[3])
        + (np.power((par[0] - par[2]), 2) / par[3])
        - np.log((par[1] / par[3]))
        - 1
    ) / 2


def relative_entropy_univariate_gaussians_logscale_variance(
    par: np.ndarray,
) -> float:
    """Calculates the KL divergence of two univariate Gaussians,
    given their mean and variance.
    This version takes the exponential of the parameter corresponding
    to the variance, mostly for ease of plotting."""

    par = par.flatten()

    par[1] = np.exp(par[1])
    par[3] = np.exp(par[3])

    if not par.shape == (4,) or not par[1] > 0 or not par[3] > 0:
        raise AttributeError(
            "Parameters must be given as np.array([mu1, sigma1^2, mu2, sigma2^2])."
        )

    return (
        (par[1] / par[3])
        + (np.power((par[0] - par[2]), 2) / par[3])
        - np.log((par[1] / par[3]))
        - 1
    ) / 2


def relative_entropy_multivariate_gaussians(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
):
    """Calculates the KL divergence of two multivariate Gaussians
    given theirs means and covariance matrices. The latter have to be positive definite.
    """
    k = mu1.shape[0]

    if (
        (not mu1.shape == (k, 1))
        or (not mu2.shape == (k, 1))
        or (not sigma1.shape == (k, k))
        or (not sigma2.shape == (k, k))
    ):
        raise AttributeError("Inconsistent dimensions.")

    if np.linalg.det(sigma1) > 0 and np.linalg.det(sigma2) > 0:
        return (
            np.trace(np.dot(np.linalg.inv(sigma2), sigma1))
            + np.dot(
                np.transpose(mu2 - mu1),
                np.dot(np.linalg.inv(sigma2), (mu2 - mu1)),
            )
            + np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
            - k
        ).item() / 2
    else:
        return np.nan


def relative_entropy_multivariate_gaussians_unified(
    par: np.ndarray,
):
    """Calculates the KL divergence of two multivariate Gaussians.

    This version takes a flat array and then constructs
    the means and covariance matrices from it, the latter should be positive definite.
    """
    par = par.flatten()

    l_par = len(par)

    found_k = False
    for k in range(int(np.ceil(np.sqrt(l_par / 2)))):
        if l_par / 2 == k * (1 + k):
            found_k = True
            break

    if found_k is False:
        raise AttributeError(
            "Invalid number of entries for two multivariate Gaussians."
        )

    mu1 = par[0:k].reshape((k, 1))
    sigma1 = par[k : (k + k * k)].reshape((k, k))
    mu2 = par[(k + k * k) : (2 * k + k * k)].reshape((k, 1))
    sigma2 = par[(2 * k + k * k) : (2 * k + 2 * k * k)].reshape((k, k))

    return relative_entropy_multivariate_gaussians(mu1, sigma1, mu2, sigma2)


# the following are building blocks for a landscape corresponding to the maxcut problem
# of graphs using a quantum circuit


def qaoa_circuit(
    n_vertices: int,
) -> tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    vertices_r = QuantumRegister(n_vertices, name="v")
    classic_r = ClassicalRegister(n_vertices, name="output")
    qc = QuantumCircuit(vertices_r, classic_r)
    return qc, vertices_r, classic_r


def initialize_circuit(
    qc: QuantumCircuit, vertices_r: QuantumRegister
) -> QuantumCircuit:
    for i in range(len(vertices_r)):
        qc.h(vertices_r[i])
    return qc


def u_c(gamma, qc, vertices_r, w):
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            if w[i, j] != 0:
                qc.cx(vertices_r[i], vertices_r[j])
                qc.rz(gamma * w[i, j], vertices_r[j])
                qc.cx(vertices_r[i], vertices_r[j])
    return qc


def u_m(beta, qc, vertices_r):
    for i in range(len(vertices_r)):
        qc.rx(2 * beta, vertices_r[i])
    return qc


def qaoa_circuit_with_layers(w, gamma_list, beta_list):
    n_vertices = len(w)
    qc, vertices_r, classic_r = qaoa_circuit(n_vertices)
    qc = initialize_circuit(qc, vertices_r)
    for gamma, beta in zip(gamma_list, beta_list):
        qc = u_c(gamma, qc, vertices_r, w)
        qc.barrier()
        qc = u_m(beta, qc, vertices_r)
    qc.measure(vertices_r, classic_r)
    return qc


def maxcut_landscape_qaoa(parameters: list, w: np.ndarray) -> float:
    """Evaluate the MaxCut objective function for given parameters
    using a qoao circuit.

    Input:

        parameters: list of [gamma_1, ... , gamma_p, beta_q, ... ,beta_p]
            that defines the angles used in the trotterized adiabatic process
            of qaoa

        w: weight matrix of a graph

    Output:

        negative of the average cut value obtained by the given qaoa parameters

    """

    nodes = w.shape[0]
    p = len(parameters) // 2
    gamma_list = parameters[:p]
    beta_list = parameters[p:]
    qc = qaoa_circuit_with_layers(w, gamma_list, beta_list)

    backend = BasicProvider().get_backend("basic_simulator")
    job = backend.run(qc, shots=1024)
    counts = job.result().get_counts()

    obj = 0
    for bitstring, count in counts.items():
        z = [1 if bit == "1" else -1 for bit in bitstring]
        obj += (
            sum(
                w[i, j] * (1 - z[i] * z[j]) / 2
                for i in range(nodes)
                for j in range(i + 1, nodes)
            )
            * count
        )

    return -obj / 1024
