import imp
import math

import numpy as np

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar2

# Utility functions to initialize the problem
from odp.Grid import Grid

# Plot options
from odp.Plots import PlotOptions, visualize_plots
from odp.Shapes import *

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray

# STUDENT CODE START
# -------------------------------
# Problem setup for Part 2
# -------------------------------

# State ordering: [px, py, theta]
grid_min = np.array([0.0, -4.0, -math.pi])
grid_max = np.array([8.0,  4.0,  math.pi])
dims = 3
N = np.array([100, 100, 72])   # recommended in the assignment
pd = [2]                       # theta is periodic
g = Grid(grid_min, grid_max, dims, N, pd)

# Obstacle implicit surface:
# l(x) = distance((px,py), c) - (r_obs + d_safe)
# collision/unsafe when l(x) <= 0
obs_center = np.array([4.3, 0.15, 0.0])
obs_radius = 0.50
d_safe = 0.20
Initial_value_f = CylinderShape(g, [2], obs_center, obs_radius + d_safe)

# Time horizon
lookback_length = 8.0
t_step = 0.05
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# Dubins car dynamics
# Same setup as earlier parts, except speed bound is now 0.5 <= v <= 1.0
my_car = DubinsCar2(
    uMin=[0.5, -1.2],
    uMax=[1.0,  1.2],
    dMax=[0.0, 0.0, 0.0],
    uMode="max",
    dMode="min",
)

# Compute backward reachable tube
compMethods = {"TargetSetMode": "minVWithV0"}
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, saveAllTimeSteps=True)

# Final value function used later by the safety filter
last_time_step_result = result[..., 0]

# Spatial derivatives
x_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=1, accuracy="low")
y_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=2, accuracy="low")
theta_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=3, accuracy="low")

# Save for later questions
np.save("hj_value_function.npy", last_time_step_result)

np.savez(
    "hj_grid.npz",
    grid_min=g.min,
    grid_max=g.max,
    pts_each_dim=g.pts_each_dim,
    periodic_dims=np.array(g.pDim),
)

np.savez(
    "hj_spatial_derivatives.npz",
    x_derivative=x_derivative,
    y_derivative=y_derivative,
    theta_derivative=theta_derivative,
)

# Full 3D zero level-set plot
po_full = PlotOptions(
    do_plot=False,
    plot_type="set",
    plotDims=[0, 1, 2],
    slicesCut=[],
    save_fig=True,
    filename="test_full.png",
)
visualize_plots(last_time_step_result, g, po_full)

# 2D slices at theta = [-pi, -pi/2, 0, pi/2]
theta_targets = [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0]
theta_indices = [int(np.argmin(np.abs(g.grid_points[2] - th))) for th in theta_targets]

for th, idx in zip(theta_targets, theta_indices):
    po_slice = PlotOptions(
        do_plot=False,
        plot_type="set",
        plotDims=[0, 1],
        slicesCut=[idx],
        save_fig=True,
        filename=f"slice_theta_{idx}.png",
    )
    visualize_plots(last_time_step_result, g, po_slice)
# STUDENT CODE END
