import numpy as np
from itertools import product

DIMS = 50



class cam_sensor:
    def __init__(self, depth, shift, fov=np.pi/2):
        self.depth = depth
        self.shift = shift
        self.fov = fov
        self.all_obs = []
        self.already_observed = []

    def get_slice_observation(self, start_grid_coord, depth, facing, max_pred=None):
        obs_idx = []

        # Bounding for env boundaries
        i_min = start_grid_coord[0] - depth
        j_min = start_grid_coord[1] - depth
        k_min = start_grid_coord[2] - depth

        if k_min < 0:
            k_min = 0
        if i_min < 0:
            i_min = 0
        if j_min < 0:
            j_min = 0

        k = range(k_min, start_grid_coord[2] + depth + 1)

        if facing == 'R':
            i = range(start_grid_coord[0] + int(2*depth/3), start_grid_coord[0] + depth + 1)
            j = range(j_min, start_grid_coord[1] + depth + 1)
        elif facing == 'L':
            if i_min - 1 > start_grid_coord[0] - int(2*depth/3):
                i = range(start_grid_coord[0] - int(2*depth/3), i_min - 1)
            else:
                i = range(i_min - 1, start_grid_coord[0] - int(2*depth/3))
            j = range(j_min, start_grid_coord[1] + depth + 1)
        elif facing =='F':
            i = range(i_min, start_grid_coord[0] + depth + 1)
            j = range(start_grid_coord[1] + int(2*depth/3), start_grid_coord[1] + depth + 1)
        elif facing == 'B':
            i = range(i_min, start_grid_coord[0] + depth + 1)
            if j_min - 1 > start_grid_coord[1] - int(2*depth/3):
                j = range(start_grid_coord[1] - int(2*depth/3), j_min - 1)
            else:
                j = range( j_min - 1, start_grid_coord[1] - int(2*depth/3))

        ijk = product(i, j, k)

        for [i, j, k] in ijk:
            if i > 0 and j > 0 and k > 0:
                if max_pred is None:
                    if i < DIMS and j < DIMS and k < DIMS:
                        obs_idx.append([i, j, k])
                else:
                    if i < max_pred and j < max_pred and k < max_pred:
                        obs_idx.append([i, j, k])
        return obs_idx

    def get_frustum_observation(self, grid_idx, facing, depth, max_pred):
        obs_idx = []

        for d in range(depth):
            idx = self.get_slice_observation(grid_idx, d, facing, max_pred)
            for i in idx:
                if i not in obs_idx:
                    obs_idx.append(i)

        return obs_idx

    def get_utility(self, grid_idx, ground_truth, facing):
        self.observed_indices = self.get_frustum_observation(grid_idx, facing, self.depth, None)

        observed_obstacles = []
        observed_fruits = []
        utility = 0
        utils_rew = 0

        for [i, j, k] in self.observed_indices:
            val = ground_truth[k][i][j]
            if val == 1:
                # if chance > NOISE:
                observed_obstacles.append([i, j, k])
                tree = 1
            elif val == 2:
                utility += 1
                observed_fruits.append([i, j, k])
                if [i, j, k] not in self.already_observed:
                    self.already_observed.append([i, j, k])
                    utils_rew += 1
            self.all_obs.append([i, j, k])

        return utility, utils_rew, observed_fruits, observed_obstacles, tree