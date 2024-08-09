"""This is a module for creating videos of scans of a function landscape."""

import numpy as np
from typing import Callable, Optional, Union, List
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

from .basic_scans import (
    landscape_scan_linear,
    ScanCollection,
    hessian_scan,
    collective_scan_linear,
)
from .dr_tools import AffineSubspace, subspace_projection, numeric_hessian


class VideoScan:
    """This object contains data corresponding to linear scans
    that form the frames of a video.

        Attributes:

            result: a grid of values sampled from a real valued function.
                the last dimension enumerates the frames

            subspaces: A list of affine linear subspaces
                along which the samples were taken.

            scope: tells you how far the subspace was scanned in each direction.
                shape is (number of directions, 2)

            func: optional attribute to record the function that was sampled.


        Methods:

            animate: creates and plays the actual video based on the results.
                this can take quite long
    """

    def __init__(
        self,
        result: np.ndarray,
        subspaces: List[AffineSubspace],
        scope: np.ndarray,  # shape (number of directions, 2). records how far
        # the landscape was scanned in each direction
        func: Optional[Callable[[np.ndarray], float]] = None,
        trajectory: Optional[np.ndarray] = None,
    ):
        self.result = result
        self.subspaces = subspaces
        self.scope = scope
        self.func = func
        self.trajectory = trajectory

    def animate(
        self,
        show_trajectory=True,
        trajectory_fade=True,
        trajectory_color="red",
        **plot_kwargs,
    ):
        """This method creates an animation of the video.
        By default this includes projections of the remaining trajectory onto each
        frame.

        Input:
            - show_trajectory: whether to include the lineplot
                that is the remaining trajectory
            - trajectory_fade: wheter to have the line fade
                towards the end of the trajectory
            - trajectory_color: the color of the lineplot

        Output:
            None, it immediately opens a video player when used in a notebook

        """

        plt.rcParams["animation.html"] = "jshtml"
        plt.ioff()

        fig, ax = plt.subplots()

        def create_animation(t):
            plt.cla()
            step_num = len(self.subspaces)

            # showing the scan
            plt.imshow(
                np.transpose(self.result[:, :, t]),
                extent=[
                    self.scope[0, 0],
                    self.scope[0, 1],
                    self.scope[1, 0],
                    self.scope[1, 1],
                ],
                **plot_kwargs,
            )
            plt.xlabel("first scan direction")
            plt.ylabel("second scan direction")

            if show_trajectory is True:
                # getting the projected trajectory
                coeff = np.zeros([step_num - t, 2])
                for step_id in range(t, step_num):

                    coeff[step_id - t, :] = np.linalg.lstsq(
                        self.subspaces[t].directions.transpose(),
                        (
                            self.trajectory[step_id, :] - self.subspaces[t].center
                        ).transpose(),
                        rcond=None,
                    )[0].flatten()

                x = coeff[:, 0]
                y = coeff[:, 1]
                reststeps = x.shape[0]

                colors = [mcolors.to_rgba(trajectory_color)] * reststeps
                if trajectory_fade is True:
                    fade = (
                        1
                        + np.cos(
                            np.pi
                            * np.linspace(0, reststeps - 1, reststeps)
                            / (reststeps)
                        )
                    ) / 2
                    for step_id in range(reststeps):
                        step_color = list(colors[step_id])
                        step_color[3] = fade[step_id]
                        colors[step_id] = tuple(step_color)

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=colors)
                ax.add_collection(lc, autolim=False)

        frames = len(self.subspaces)
        ani = animation.FuncAnimation(fig, create_animation, frames=frames)

        return ani


class VideoCollectiveScan:
    """This object contains data corresponding to collective scans
    that form the frames of a video.

        Attributes:

            result: a grid of values sampled from a real valued function.
                the last dimension enumerates the frames

            subspaces: A list of affine linear subspaces
                along which the samples were taken.

            scope: tells you how far the subspace was scanned in each direction.
                shape is (number of directions, 2)

            func: optional attribute to record the function that was sampled.


        Methods:

            animate: creates and plays the actual video based on the results.
                this can take quite long
    """

    def __init__(
        self,
        scans: List[ScanCollection],
        scope: np.ndarray,  # shape (number of directions, 2). records how far
        # the landscape was scanned in each direction
        func: Optional[Callable[[np.ndarray], float]] = None,
        trajectory: Optional[np.ndarray] = None,
    ):
        self.scans = scans
        self.scope = scope
        self.func = func
        self.trajectory = trajectory

    def animate(self, **plot_kwargs):
        """This method creates an animation of the video.
        By default this includes projections of the remaining trajectory onto each
        frame.

        Input:
            - show_trajectory: whether to include the lineplot
                that is the remaining trajectory
            - trajectory_fade: wheter to have the line fade
                towards the end of the trajectory
            - trajectory_color: the color of the lineplot

        Output:
            None, it immediately opens a video player when used in a notebook

        """

        plt.rcParams["animation.html"] = "jshtml"
        plt.ioff()

        fig, ax = plt.subplots()

        def create_animation(t):
            plt.cla()
            # step_num = len(self.scans)

            # showing the scan
            self.scans[t].show(**plot_kwargs)

            ax = plt.gca()
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            # plt.imshow(
            #     np.transpose(self.result[:, :, t]),
            #     extent=[
            #         self.scope[0, 0],
            #         self.scope[0, 1],
            #         self.scope[1, 0],
            #         self.scope[1, 1],
            #     ],
            #     **plot_kwargs
            # )
            # plt.xlabel("first scan direction")
            # plt.ylabel("second scan direction")

        frames = len(self.scans)
        ani = animation.FuncAnimation(fig, create_animation, frames=frames)

        return ani


def trajectory_scan_stepwise_pca(
    func: Callable[[np.ndarray], float],
    trajectory: np.ndarray,
    scope: Union[np.ndarray, float] = 5,
    resolution: Optional[Union[int, np.ndarray]] = 10,
    pools: Optional[int] = 1,
) -> VideoScan:
    """The purpose of this function is to scan a landscape defined by func
    along the path defined by trajectory. For each step in the trajectory it will
    project the remaining trajectory onto the orthogonal complement of the gradient
    of the previous step and then do pca on that, to get the scan directions.

    The first and the last frame show the pca of the entire trajectory without any
    projections because above described method cannot be applied there.

    Input:

        func: The function that defines the landscape.

        trajectory: an array of points in the lanscape
            must be of shape (number of steps, dimension of superspace)

        scope: How far to scan in each direction of subspace.
        The size has to be (d_num,2), where for each direction the first entry is the
        beginning of the scope and the second is the end.
        If provided as float, the beginning will be -scope and the end +scope.

        resolution: How many samples to take in each direction of subspace.
        If provided as int, the resolution will be the same for each direction.

    Output:

        A trajectory scan object.

    """

    d_num = 2
    d_dim = trajectory.shape[1]

    if not isinstance(scope, np.ndarray):
        scope = scope * np.append(-np.ones([d_num, 1]), np.ones([d_num, 1]), 1)

    if isinstance(resolution, int):
        resolution = resolution * np.ones(d_num, dtype=int)

    traj_length = trajectory.shape[0]
    result = np.zeros(shape=np.append(resolution, traj_length))
    subspaces = [None] * traj_length
    directions = np.zeros([d_num, d_dim])

    # doing the first scan separately because there is no preceding step to define the
    # direction. instead it will just show the pca of the whole trajectory
    step_id = 0

    directions = PCA(n_components=2).fit(trajectory).components_
    directions = np.dot(np.dot(directions, directions.transpose()), directions)

    center = PCA(n_components=2).fit(trajectory).mean_.reshape([1, d_dim])

    subspace = AffineSubspace(directions=directions, center=center)

    scan = landscape_scan_linear(func, subspace, scope, resolution, pools=pools)
    result[:, :, step_id] = scan.result
    subspaces[step_id] = subspace

    # doing the stepwise pca scan for the remaining steps
    for step_id in range(1, traj_length - 1):
        stepspace = AffineSubspace(
            directions=np.array(
                [
                    (trajectory[step_id, :] - trajectory[step_id - 1, :])
                    / np.linalg.norm(
                        trajectory[step_id, :] - trajectory[step_id - 1, :]
                    )
                ]
            ),
            center=np.array([trajectory[step_id, :]]),
        )
        cloud = subspace_projection(
            trajectory[step_id:, :], stepspace, orthogonal=True, relative=True
        )
        directions_old = directions
        directions = PCA(n_components=2).fit(cloud).components_
        directions = np.dot(np.dot(directions, directions.transpose()), directions)
        # now making sure that directions dont 'flip sign' by matching them with
        for d_id in range(d_num):
            if np.linalg.norm(
                directions[d_id, :] - directions_old[d_id, :]
            ) > np.linalg.norm(-directions[d_id, :] - directions_old[d_id, :]):
                directions[d_id, :] = -directions[d_id, :]

        subspace = AffineSubspace(
            directions=directions, center=np.array([trajectory[step_id, :]])
        )
        scan = landscape_scan_linear(func, subspace, scope, resolution, pools=pools)

        result[:, :, step_id] = scan.result
        subspaces[step_id] = subspace

    # doing the last scan separately because there are not enough following points
    # to do pca on, instead it will just show the pca of the whole trajectory
    step_id = traj_length - 1

    directions = PCA(n_components=2).fit(trajectory).components_
    directions = np.dot(np.dot(directions, directions.transpose()), directions)

    center = PCA(n_components=2).fit(trajectory).mean_.reshape([1, d_dim])

    subspace = AffineSubspace(directions=directions, center=center)

    scan = landscape_scan_linear(func, subspace, scope, resolution, pools=pools)
    result[:, :, step_id] = scan.result
    subspaces[step_id] = subspace

    return VideoScan(
        result=result,
        subspaces=subspaces,
        scope=scope,
        func=func,
        trajectory=trajectory,
    )


def trajectory_scan_stepwise_hessian(
    func: Callable[[np.ndarray], float],
    trajectory: np.ndarray,
    scope: Union[np.ndarray, float] = 5,
    resolution: Optional[Union[int, np.ndarray]] = 10,
    epsilon: Optional[float] = 0.01,
    sharpness: Optional[int] = 2,
    pools: Optional[int] = 1,
) -> VideoCollectiveScan:
    """This function creates one frame per point in the trajectory by performing
    a hessian_scan on the orthogonal complement of the
    direction of the next step.
    If the latter is the same a the gradient, then this is related to the
    second fundamental form.

    Note: since this cannot be done with the last frame,
    it shows the hessian scan on the full space.

    Input:

        func: The function that defines the landscape.

        trajectory: an array of points in the lanscape
            must be of shape (number of steps, dimension of superspace)

        scope: How far to scan in each direction of subspace.
        The size has to be (d_num,2), where for each direction the first entry is the
        beginning of the scope and the second is the end.
        If provided as float, the beginning will be -scope and the end +scope.

        resolution: How many samples to take in each direction of subspace.
        If provided as int, the resolution will be the same for each direction.

    Output:

        A trajectory scan object.

    """

    d_dim = trajectory.shape[1]
    d_num = d_dim - 1

    if not isinstance(scope, np.ndarray):
        scope = scope * np.append(-np.ones([d_num, 1]), np.ones([d_num, 1]), 1)

    if isinstance(resolution, int):
        resolution = resolution * np.ones(d_num, dtype=int)

    traj_length = trajectory.shape[0]

    # result = np.zeros(shape=np.append(resolution, traj_length))
    scans = [None] * traj_length
    directions = np.zeros([d_num, d_dim])

    # doing the stepwise pca scan for the remaining steps
    for step_id in range(0, traj_length - 1):

        # getting direction of next step
        stepspace = AffineSubspace(
            directions=np.array(
                [
                    (trajectory[step_id + 1, :] - trajectory[step_id, :])
                    / np.linalg.norm(
                        trajectory[step_id + 1, :] - trajectory[step_id, :]
                    )
                ]
            ),
            center=np.array([trajectory[step_id, :]]),
            sharpness=sharpness,
        )

        stepspace_oc = stepspace.orth()
        H = numeric_hessian(func=func, subspace=stepspace_oc, epsilon=epsilon)
        H.calc_evs()

        directions_old = directions
        directions = np.dot(H.eigenvectors.transpose(), stepspace_oc.directions)
        # directions = stepspace_oc.directions
        # now making sure that directions dont 'flip sign' by matching them with
        # the previous step
        for d_id in range(d_num):
            if np.linalg.norm(
                directions[d_id, :] - directions_old[d_id, :]
            ) > np.linalg.norm(-directions[d_id, :] - directions_old[d_id, :]):
                directions[d_id, :] = -directions[d_id, :]

        stepspace_h = AffineSubspace(
            directions=directions, center=stepspace_oc.center, orthonormalize=False
        )

        scans[step_id] = collective_scan_linear(
            func=func,
            subspace=stepspace_h,
            scope=scope,
            resolution=resolution,
            pools=pools,
        )

    # doing the last step separately
    step_id = traj_length - 1
    finalspace = AffineSubspace(
        directions=np.eye(d_dim),
        center=trajectory[[step_id], :],
        sharpness=stepspace.sharpness,
    )
    scans[step_id] = hessian_scan(
        func=func,
        subspace=finalspace,
        scope=np.append(scope, scope[[d_dim - 2], :], axis=0),
        resolution=np.append(resolution, resolution[d_dim - 2]),
        epsilon=epsilon,
        pools=pools,
    )[0]

    return VideoCollectiveScan(
        scans=scans, scope=scope, func=func, trajectory=trajectory
    )
