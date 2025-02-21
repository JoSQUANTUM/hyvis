"""This is a module for creating videos of scans of a function landscape."""

from typing import Callable, List, Optional, Union
from warnings import warn

import numpy as np
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from .basic_scans import (
    ScanCollection,
    collective_scan_linear,
    hessian_scan,
    landscape_scan_linear,
)
from .dr_tools import AffineSubspace, numeric_hessian, subspace_projection


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
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        show_trajectory: Optional[bool] = True,
        trajectory_fade: Optional[bool] = True,
        trajectory_color: Optional[str] = "red",
        **plot_kwargs,
    ):
        """This method creates an animation of the video.
        By default this includes projections of the remaining trajectory onto each
        frame. Note that if the range for the colors is not fixed (e.b. by setting
        vmin and vmax in plot_kwargs) then each frame will be colored separately.

        Input:
            - show_trajectory: whether to include the lineplot
                that is the remaining trajectory
            - trajectory_fade: wheter to have the line fade
                towards the end of the trajectory
            - trajectory_color: the color of the lineplot

        Output:
            None, it immediately opens a video player when used in a notebook

        """

        if "vmin" not in plot_kwargs or "vmax" not in plot_kwargs:
            warn(
                """The range of the colormap has not been specified, each frame will be colored individually. To get consistent coloring specify vmin and vmax as arguments for this method."""
            )

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
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            if self.trajectory is not None:
                if show_trajectory is True:
                    # getting the projected trajectory
                    coeff = np.zeros([step_num - t, 2])
                    for step_id in range(t, step_num):
                        coeff[step_id - t, :] = np.linalg.lstsq(
                            self.subspaces[t].directions.transpose(),
                            (
                                self.trajectory[step_id, :]
                                - self.subspaces[t].center
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
                    segments = np.concatenate(
                        [points[:-1], points[1:]], axis=1
                    )
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

        frames = len(self.scans)
        ani = animation.FuncAnimation(fig, create_animation, frames=frames)

        return ani


def volume_scan(
    func: Callable[[np.ndarray], float],
    subspace: AffineSubspace,
    scope: Optional[Union[np.ndarray, float]] = 5,
    resolution: Optional[Union[int, np.ndarray]] = 10,
    pools: Optional[int] = 1,
) -> VideoScan:
    """This function creates a videoscan where each frame scans the subspace except
    in the last direction, which is instead used to move the space. For example,
    if the subspace is 3D this will create a video of 2D scans.

    The subspace should have at least 3 dimensions. It can have more, but then
    it can not be animated.
    """

    d_num = subspace.directions.shape[0]

    if not isinstance(scope, np.ndarray):
        scope = scope * np.append(-np.ones([d_num, 1]), np.ones([d_num, 1]), 1)

    if not (scope[:, 0] < scope[:, 1]).all():
        raise ValueError(
            """scope[id_d,0] must be strictly smaller than scope[id_d,1] for each
            direction."""
        )

    if np.isscalar(resolution):
        resolution = resolution * np.ones(d_num, dtype=int)

    scanres = resolution[-1]
    subspaces = []
    for i_scan in range(scanres):
        center = np.array(
            [
                (
                    subspace.center.flatten()
                    + scope[-1, 0] * subspace.directions[-1, :]
                    + (-scope[-1, 0] + scope[-1, 1])
                    * i_scan
                    * ((-scope[-1, 0] + scope[-1, 1]) / (scanres - 1))
                    * subspace.directions[-1, :]
                )
            ]
        )
        subspaces.append(
            AffineSubspace(subspace.directions[:-1, :], center=center)
        )

    scan = landscape_scan_linear(
        func=func,
        subspace=subspace,
        scope=scope,
        resolution=resolution,
        pools=pools,
    )

    return VideoScan(
        result=scan.result,
        subspaces=subspaces,
        scope=scan.scope,
        func=func,
    )


def operator_scan(
    func: Callable[[np.ndarray], float],
    subspace: AffineSubspace,
    operator: Optional[np.ndarray] = None,
    shift: Optional[np.ndarray] = None,
    steps: Optional[int] = 10,
    scope: Optional[Union[np.ndarray, float]] = 5,
    resolution: Optional[Union[int, np.ndarray]] = 10,
    pools: Optional[int] = 1,
) -> VideoScan:
    """performs some number of scans, where in each step the
    subspace is transformed by an affine linear operator"""

    d_dim = subspace.directions.shape[1]
    d_num = subspace.directions.shape[0]

    if not isinstance(scope, np.ndarray):
        scope = scope * np.append(-np.ones([d_num, 1]), np.ones([d_num, 1]), 1)

    if not (scope[:, 0] < scope[:, 1]).all():
        raise ValueError(
            """scope[id_d,0] must be strictly smaller than scope[id_d,1] for each
            direction."""
        )

    if np.isscalar(resolution):
        resolution = resolution * np.ones(d_num, dtype=int)

    if operator is None:
        operator = np.eye(d_dim)
    if shift is None:
        shift = np.zeros(shape=subspace.center.shape)

    result = np.zeros(shape=np.append(resolution, steps))

    subspaces = []
    for i_step in range(steps):
        scan = landscape_scan_linear(
            func, subspace, scope, resolution, pools=pools
        )
        subspaces.append(subspace)
        result[..., i_step] = scan.result

        directions = (operator @ subspace.directions.T).T
        center = subspace.center + shift
        subspace = AffineSubspace(directions=directions, center=center)

    return VideoScan(
        result=result,
        subspaces=subspaces,
        scope=scope,
        func=func,
    )


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

    scan = landscape_scan_linear(
        func, subspace, scope, resolution, pools=pools
    )
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
        directions = np.dot(
            np.dot(directions, directions.transpose()), directions
        )
        # now making sure that directions dont 'flip sign' by matching them with
        for d_id in range(d_num):
            if np.linalg.norm(
                directions[d_id, :] - directions_old[d_id, :]
            ) > np.linalg.norm(-directions[d_id, :] - directions_old[d_id, :]):
                directions[d_id, :] = -directions[d_id, :]

        subspace = AffineSubspace(
            directions=directions, center=np.array([trajectory[step_id, :]])
        )
        scan = landscape_scan_linear(
            func, subspace, scope, resolution, pools=pools
        )

        result[:, :, step_id] = scan.result
        subspaces[step_id] = subspace

    # doing the last scan separately because there are not enough following points
    # to do pca on, instead it will just show the pca of the whole trajectory
    step_id = traj_length - 1

    directions = PCA(n_components=2).fit(trajectory).components_
    directions = np.dot(np.dot(directions, directions.transpose()), directions)

    center = PCA(n_components=2).fit(trajectory).mean_.reshape([1, d_dim])

    subspace = AffineSubspace(directions=directions, center=center)

    scan = landscape_scan_linear(
        func, subspace, scope, resolution, pools=pools
    )
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

        trajectory: an array of points in the landscape
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
        directions = np.dot(
            H.eigenvectors.transpose(), stepspace_oc.directions
        )
        # directions = stepspace_oc.directions
        # now making sure that directions dont 'flip sign' by matching them with
        # the previous step
        for d_id in range(d_num):
            if np.linalg.norm(
                directions[d_id, :] - directions_old[d_id, :]
            ) > np.linalg.norm(-directions[d_id, :] - directions_old[d_id, :]):
                directions[d_id, :] = -directions[d_id, :]

        stepspace_h = AffineSubspace(
            directions=directions,
            center=stepspace_oc.center,
            orthonormalize=False,
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
