import numpy as np



class cam_sensor:
    def __init__(self, fruit_grids, depth, shift, dim, fov=np.pi/2):
        self.dim = dim
        self.depth = depth
        self.shift = shift
        self.fov = fov
        self.already_observed = []
        self.fruit_grids = fruit_grids

    def get_slice_observation(self, start_grid_coord, depth, facing, max_pred=None):
        # Compute lower boundaries, clamped at 0
        i_min = max(0, start_grid_coord[0] - depth)
        j_min = max(0, start_grid_coord[1] - depth)
        k_min = max(0, start_grid_coord[2] - depth)
        k_max = start_grid_coord[2] + depth + 1  # Upper bound (non-inclusive)

        # Create the k-axis values
        k_vals = np.arange(k_min, k_max)

        # Determine the i and j ranges based on the facing direction
        if facing == 'R':
            i_start = start_grid_coord[0] + int(2 * depth / 3)
            i_end = start_grid_coord[0] + depth + 1
            i_vals = np.arange(i_start, i_end)
            j_vals = np.arange(j_min, start_grid_coord[1] + depth + 1)
        elif facing == 'L':
            # Determine the correct ordering for the left-facing range
            i_lower = min(start_grid_coord[0] - int(2 * depth / 3), i_min - 1)
            i_upper = max(start_grid_coord[0] - int(2 * depth / 3), i_min - 1)
            i_vals = np.arange(i_lower, i_upper)
            j_vals = np.arange(j_min, start_grid_coord[1] + depth + 1)
        elif facing == 'F':
            i_vals = np.arange(i_min, start_grid_coord[0] + depth + 1)
            j_start = start_grid_coord[1] + int(2 * depth / 3)
            j_vals = np.arange(j_start, start_grid_coord[1] + depth + 1)
        elif facing == 'B':
            i_vals = np.arange(i_min, start_grid_coord[0] + depth + 1)
            j_lower = min(start_grid_coord[1] - int(2 * depth / 3), j_min - 1)
            j_upper = max(start_grid_coord[1] - int(2 * depth / 3), j_min - 1)
            j_vals = np.arange(j_lower, j_upper)
        else:
            raise ValueError("Facing must be one of 'R', 'L', 'F', or 'B'.")

        # Generate all combinations of i, j, k using meshgrid
        I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        coords = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=-1)

        # Filter: keep only coordinates where each component is > 0
        valid_mask = np.all(coords > 0, axis=1)
        coords = coords[valid_mask]

        # Apply boundary constraints: use self.dim if max_pred is None
        boundary = self.dim if max_pred is None else max_pred
        valid_mask = np.all(coords < boundary, axis=1)
        coords = coords[valid_mask]

        return coords.tolist()

    def get_frustum_observation(self, grid_idx, facing, depth, max_pred):
        obs_idx = []

        for d in range(depth):
            idx = self.get_slice_observation(grid_idx, d, facing, max_pred)
            for i in idx:
                if i not in obs_idx:
                    obs_idx.append(i)

        return obs_idx

    def get_utility_fast(self, grid_idx, facing):
        self.observed_indices = self.get_frustum_observation(grid_idx, facing, self.depth, None)

        # print(len(self.already_observed))
        observed_fruits = []
        utility = 0
        utils_rew = 0

        for [i, j, k] in self.observed_indices:
            if [i, j, k] in self.fruit_grids:
                observed_fruits.append([i, j, k])
                if [i, j, k] not in self.already_observed:
                    self.already_observed.append([i, j, k])
                    utils_rew += 1

        return utility, utils_rew, observed_fruits, []
    
    def get_utility(self, grid_idx, ground_truth, facing):
        self.observed_indices = self.get_frustum_observation(grid_idx, facing, self.depth, None)

        # print(len(self.already_observed))
        observed_fruits = []
        utility = 0
        utils_rew = 0

        for [i, j, k] in self.observed_indices:
            val = ground_truth[k][i][j]
            if val == 2:
                utility += 1
                observed_fruits.append([i, j, k])
                if [i, j, k] not in self.already_observed:
                    self.already_observed.append([i, j, k])
                    utils_rew += 1

        return utility, utils_rew, observed_fruits, []