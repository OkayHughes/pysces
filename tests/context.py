import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_figdir():
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_figures")


def plot_grid(grid, ax):
  from matplotlib import collections as mc
  import numpy as np
  npt = grid["physical_coords"].shape[1]
  lines = []
  for i_idx in range(npt - 1):
    for j_idx in [0, npt - 1]:
      points_start = zip(grid["physical_coords"][:, i_idx, j_idx, 1].flatten(),
                         grid["physical_coords"][:, i_idx, j_idx, 0].flatten())
      points_end = zip(grid["physical_coords"][:, i_idx + 1, j_idx, 1].flatten(),
                       grid["physical_coords"][:, i_idx + 1, j_idx, 0].flatten())
      lines += zip(points_start, points_end)
  for j_idx in range(npt - 1):
    for i_idx in [0, npt - 1]:
      points_start = zip(grid["physical_coords"][:, i_idx, j_idx, 1].flatten(),
                         grid["physical_coords"][:, i_idx, j_idx, 0].flatten())
      points_end = zip(grid["physical_coords"][:, i_idx, j_idx + 1, 1].flatten(),
                       grid["physical_coords"][:, i_idx, j_idx + 1, 0].flatten())
      lines += zip(points_start, points_end)
  lines = list(filter(lambda line: np.abs(line[1][0] - line[0][0]) < np.pi, lines))
  lc = mc.LineCollection(lines, colors="k", alpha=0.5, linewidths=.05)
  ax.add_collection(lc)


extensive = False
test_division_factor = 1.0 if extensive else 1000.0
test_npts = [3, 4, 5, 6] if extensive else [3, 4]

seed = 0
