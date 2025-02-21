"""This module contains tools for dimensionaly reduction and geomerty."""

import numpy as np
from typing import Optional, Callable, List
import copy
from matplotlib import pyplot as plt


class AffineSubspace:
    """This object contains data defining an affine linear subspace of some higher
      dimensional real or complex vector space.
    It consists of an orthonormal basis of directions
    and an arbitary vector as the center.

    Attribues:

        directions: An approximate orthonormal set of vectors from some high
        dimensional superspace.
        Must be of shape (number of directions, dimension of superspace)

        center: A vector from the same space as the directions.

        sharpness: The decimals error allowed in the check for orthonormality.
        That means both in terms of norm and orthogonality of each pair of directions.

    """

    def __init__(
        self,
        directions: np.ndarray,
        center: Optional[np.ndarray] = None,
        sharpness: Optional[int] = 2,
        orthonormalize: Optional[bool] = True,
    ):
        d_num = directions.shape[0]
        d_dim = directions.shape[1]

        self.sharpness = sharpness

        # checking orthonormality and enforcing it if necessary
        if (
            orthonormalize is False
            or (
                np.round(np.dot(directions, np.transpose(directions)), sharpness)
                == np.identity(d_num, dtype=float)
            ).all()
        ):
            self.directions = directions
        else:
            # warnings.warn(
            #     """Directions were not given as orthonormal basis so they will be
            #     adjusted automatically."""
            # )
            self.directions = gramschmidt(directions.transpose()).transpose()
            if not (
                np.round(
                    np.dot(self.directions, np.transpose(self.directions)), sharpness
                )
                == np.identity(d_num, dtype=float)
            ).all():
                raise AttributeError(
                    "Provided directions are not linearly independent."
                )

        # checking center dimension, or setting it to 0 if none was provided
        if center is None:
            self.center = np.zeros((1, d_dim))
        else:
            if center.shape == (1, d_dim):
                self.center = center
            else:
                raise AttributeError(
                    "Center dimension must be the same as direction dimension."
                )

    def orth(self):
        """This method returns the orthogonal complement of the AffineSubspace.
        Note that this forgets everything about the orientation of the directions,
        in particular, this operation is not self inverse regarding the exact basis
        """
        d_dim = self.directions.shape[1]

        candidates = subspace_projection(
            pointcloud=np.eye(d_dim) + np.tile(self.center, [d_dim, 1]),
            target_space=self,
            orthogonal=True,
            relative=True,
        )

        i_c = 0
        found_start = False
        while not found_start:
            if np.linalg.norm(candidates[[i_c], :]) < (
                1 / np.power(10, self.sharpness)
            ):
                i_c += 1
            else:
                OC = AffineSubspace(
                    candidates[[i_c], :], center=self.center, sharpness=self.sharpness
                )
                found_start = True

        while (
            not OC.directions.shape[0]
            == self.directions.shape[1] - self.directions.shape[0]
        ):
            i_c += 1

            if i_c == d_dim:
                raise AttributeError(
                    "Did not find enough linear independent candidates."
                )

            if not np.linalg.norm(
                subspace_projection(
                    pointcloud=candidates[[i_c], :] + self.center,
                    target_space=OC,
                    orthogonal=True,
                    relative=True,
                )
            ) < (1 / np.power(10, self.sharpness)):
                OC = AffineSubspace(
                    np.append(OC.directions, candidates[[i_c], :], axis=0),
                    center=OC.center,
                    sharpness=OC.sharpness,
                )

        if (
            not OC.directions.shape[0]
            == self.directions.shape[1] - self.directions.shape[0]
        ):
            raise AttributeError("Dimension of complement is not as expected.")
        else:
            return OC


class Hessian:
    """This is a class for hessian matrices evaluated at specific points.

    Attributes:

        matrix: the hessian matrix itself

        func: the function for which it was calculated

        subspace: the point at which it was calculated and
         the directions in which the partial derivatives were taken

        epsilon: the offset used in the numeric calculation

    Methods:

        calc_evs: calculates eigenvectors and eigenvalues of the hessian matrix

        show_evs: creates a scatterplot of the eigenvalues, lowest to highest
    """

    def __init__(
        self,
        matrix: np.ndarray,
        func: Optional[Callable[[np.ndarray], float]] = None,
        subspace: Optional[AffineSubspace] = None,
        epsilon: Optional[float] = None,
    ):
        self.matrix = matrix
        self.func = func
        self.subspace = subspace
        self.epsilon = epsilon

    def calc_evs(
        self,
    ):
        """Calculates and adds the eigenvalues and eigenvectors
        as attributes to the object. Sorts eigenvalues lowest to highest."""
        if not hasattr(self, "eigenvalues"):
            ev = np.linalg.eigh(self.matrix)

            # sorting the eigenvalues in ascending order
            order = np.argsort(ev.eigenvalues)

            self.eigenvalues = ev.eigenvalues[order]
            self.eigenvectors = ev.eigenvectors[:, order]

    def show_evs(self, **plot_kwargs):
        """Creates a scatterplot of the eigenvalues. y-axis is the actual eigenvalue,
        x-axis is the index of each eigenvalue in the order lowest to highest"""
        if not hasattr(self, "eigenvalues"):
            self.calc_evs()

        plt.scatter(range(1, self.matrix.shape[0] + 1), self.eigenvalues, **plot_kwargs)
        plt.xticks(
            ticks=range(1, self.matrix.shape[0] + 1),
            labels=range(1, self.matrix.shape[0] + 1),
        )
        plt.xlabel("index")
        plt.ylabel("eigenvalue")


def subspace_projection(
    pointcloud: np.ndarray,
    target_space: AffineSubspace,
    orthogonal: Optional[bool] = False,
    relative: Optional[bool] = False,
) -> np.ndarray:
    """This function takes as input a pointcloud and an affine linear subspace
    and puts out the projection of the pointcloud onto that subspace, or its orthogonal
    complement.

    Please note that I believe this is the most accurate way to to this in numpy but it
    is still not perfect. Especially when you involve irraltional entries
    like np.sqrt(2) or np.pi you will likely get some inaccurary.

    IMPORTANTLY it uses the fact that the directions of AffineSubspace are orthonormal.

    Input:

        - pointcloud: an arbitrary collection with same dimension as the superspace,
            ie shape (number of points, dimension of superspace)
        - target_space: the subspace that defines the projection
        - orthogonal: whether to project to target_space (False) or
            its orthogonal complement (True)
        - relative: whether to represent the output relative
            to zero (False) or to the center of the AffineSubspace (True)

    Output:

        - np.ndarray with same shape as pointcloud, representing its projection

    """

    if pointcloud.shape[1] != target_space.directions.shape[1]:
        raise AttributeError(
            """The shape of pointclould must be (number of points, dimension of
            superspace) even if its just one point."""
        )

    d_num = target_space.directions.shape[0]
    p_num = pointcloud.shape[0]
    points_relative = pointcloud - np.repeat(target_space.center, p_num, axis=0)

    # calculating the coefficients that each element of the subspace basis
    # would have in a linear combination for each point,
    # if it was extended to a full orthonormal basis of the space
    coeff = np.zeros([d_num, p_num])
    for i_dir in range(d_num):
        coeff[i_dir] = np.dot(
            points_relative, target_space.directions[i_dir, :].transpose()
        )

    # returning the relevant "parts" of the points, possibly offset by the center
    if orthogonal is False:
        return np.dot(target_space.directions.transpose(), coeff).transpose() + (
            not relative
        ) * np.repeat(target_space.center, p_num, axis=0)
    elif orthogonal is True:
        return (
            points_relative
            - np.dot(target_space.directions.transpose(), coeff).transpose()
            + (not relative) * np.repeat(target_space.center, p_num, axis=0)
        )


def gramschmidt(V: np.ndarray):
    """This function just orthogonalizes a matrix V using the Gram-Schmidt algorithm.
    It turns the columns of V into an orthonormal basis/set of vectors.
    It is here because other methods, like scipy.linalg.orth
    seem to be less accurate.
    """

    U = np.zeros(shape=V.shape, dtype=float)
    U[:, 0] = V[:, 0] / np.linalg.norm(V[:, 0])
    for i in range(1, V.shape[1]):
        U[:, i] = V[:, i]
        for j in range(i):
            U[:, i] = U[:, i] - np.dot(U[:, j].transpose(), U[:, i]) * U[:, j]
        U[:, i] = U[:, i] / np.linalg.norm(U[:, i])
    return U


def numeric_partial_derivative(
    func: Callable[[np.ndarray], float],
    pos: List[int],
    point: np.ndarray,
    epsilon: Optional[float] = 0.01,
) -> float:
    """This function approximates chains of partial derivatives using central finite
    differences around a specific point.

    Input:
        - func: the function to be differentiated
        - pos: the positions for which the partial derivatives are to be taken,
            ie the first entry of this list would be the rightmost index
            when writing it on paper
            (the entries must be between 1 and the dimension of the argument space)
        - point: the point at which the derivative is to be evaluated
        - epsilon: the offset used to calculate the numeric derivative,
            should either be very small or chosen according to some parameter shift rule
    """
    depth = len(pos)

    value_acc = 0
    # going through the binary representations of all numbers up to 2^depth
    # this encodes the nested additions and substractions in the approximation
    for i in range(np.power(2, depth)):
        i_c = copy.copy(i)
        par = copy.copy(point)
        sign = 1

        # depending on current digit, shift the parameter vector by foward or backward
        for i_d in range(depth - 1, -1, -1):

            if np.power(2, i_d) <= i_c:
                i_c = i_c - np.power(2, i_d)
                esign = -1
            else:
                esign = 1

            sign = sign * esign
            par[pos[i_d]] = par[pos[i_d]] + esign * epsilon / 2

        value_acc = value_acc + sign * func(par)

    # returning the accumulated sum of differences
    # divided by the offset for each step of derivation
    return value_acc / np.power(epsilon, depth)


def numeric_gradient(
    func: Callable[[np.ndarray], float],
    subspace: AffineSubspace,
    epsilon: Optional[float] = 0.01,
    relative: bool = False,
) -> np.ndarray:
    """This approximates the gradient of a function
    (restricted to the given subspace) unsing finite central differences.

    Input:
        - func: any function mapping an array of real numbers to a real number
        - subspace: an affine linear subspace of the parameter space,
            determines around which point to calculate the gradient
            and what basis to represent it in
        - epsilon: the offset used in the approximation, should either be small
            or chosen according to some parameter shift rule
    """

    d_num = subspace.directions.shape[0]
    grad = np.zeros((1, d_num))

    # reparametrizing the function to the subspace
    def func_sub(
        par=np.ndarray,
    ) -> float:
        return func(subspace.center + np.dot(subspace.directions.transpose(), par))

    for k in range(d_num):
        grad[0, k] = numeric_partial_derivative(
            func_sub, [k], np.zeros([d_num]), epsilon=epsilon
        )

    if relative is False:
        return grad
    else:
        return np.dot(grad, subspace.directions)


def numeric_hessian(
    func: Callable[[np.ndarray], float],
    subspace: AffineSubspace,
    epsilon: Optional[float] = 0.01,
) -> Hessian:
    """This approximates the hessian of a given function
    (restricted to the given subspace) unsing finite central differences.

    Input:
        - func: any function mapping an array of real numbers to a real number
        - subspace: an affine linear subspace of the parameter space,
            determines around which point to calculate the hessian
            and in which directions to take the partial derivatives
        - epsilon: the offset used in the approximation, should either be small
            or chosen according to some parameter shift rule
    """
    d_num = subspace.directions.shape[0]
    H = np.zeros([d_num, d_num])

    # reparametrising the function to the subspace
    def func_sub(
        par=np.ndarray,
    ) -> float:
        return func(subspace.center + np.dot(subspace.directions.transpose(), par))

    # getting all second order partial derivatives
    for j in range(d_num):
        for k in range(d_num):
            H[j, k] = numeric_partial_derivative(
                func_sub, [j, k], np.zeros([d_num]), epsilon=epsilon
            )

    return Hessian(matrix=H, func=func, subspace=subspace, epsilon=epsilon)
