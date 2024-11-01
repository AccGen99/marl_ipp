import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point



class Build:
    def __init__(self):
        self.centre = None
        self.top_right = None
        self.bottom_left = None
        self.grid_index = None
        self.height = None
        self.poly2D_coords = None
        self.target_coords = []

class Obstacle:
    def __init__(self, num_builds, test_name, max_height = 1.0, h_max_ratio = 0.99, dim = 50, width_cells = 3):
        self.dim = dim
        self.width = width_cells
        self.max_height = max_height * h_max_ratio
        self.num_builds = num_builds #5*5 #7*7
        self.builds = []

        if test_name == 'random':
            self.generate_random()
        elif test_name == 'grid':
            self.generate_grid()

    def get_buildings(self):
        return self.builds

    def get_3d_gt(self, fruit_coords):
        self.distrib_3d = np.zeros((self.dim, self.dim, self.dim))

        for coord in fruit_coords:
            x_idx, y_idx, z_idx = self.find_grid_idx(coord)
            self.distrib_3d[z_idx][x_idx][y_idx] = 2

        return self.distrib_3d

    def find_grid_coord(self, grid_val):
        x_coord = (grid_val[0] + 0.5) * 1.0 / self.dim
        y_coord = (grid_val[1] + 0.5) * 1.0 / self.dim
        z_coord = (grid_val[2] + 0.5) * 1.0 / self.dim

        return np.array([x_coord, y_coord, z_coord])

    def find_grid_coord_pred(self, grid_val, grid_res):
        x_coord = (grid_val[0] + 0.5) * 1.0 / grid_res
        y_coord = (grid_val[1] + 0.5) * 1.0 / grid_res
        z_coord = (grid_val[2] + 0.5) * 1.0 / grid_res

        return np.array([x_coord, y_coord, z_coord])

    def find_grid_idx_pred(self, coords, grid_res):
        index_x = math.floor(coords[0] * grid_res)# / self.max_height)
        index_y = math.floor(coords[1] * grid_res)# / self.max_height)
        index_z = math.floor(coords[2] * grid_res)# / self.max_height)
        return index_x, index_y, index_z

    def find_grid_idx(self, coords):
        index_x = math.floor(coords[0] * self.dim)# / self.max_height)
        index_y = math.floor(coords[1] * self.dim)# / self.max_height)
        index_z = math.floor(coords[2] * self.dim)# / self.max_height)
        return index_x, index_y, index_z

    def get_gt_occupancy_grid(self, target_coords, obstacles=[], prev_grid = None, is_ground_truth = False):
        '''
        Grid value legend
        0 -> Free area
        1 -> Obstacle
        2 -> Target
        '''
        if prev_grid is None:
            obs_grid = np.zeros((self.dim, self.dim, self.dim))
        else:
            obs_grid = prev_grid

        if is_ground_truth:
            for build in self.builds:
                build_poly = Polygon(build.poly2D_coords)
                height = build.height
                for i in range(self.dim):
                    for j in range(self.dim):
                        idx = [i, j, 0]
                        x, y, _ = self.find_grid_coord(idx)
                        pt = Point(x,y)
                        if pt.within(build_poly):
                            _, _, k_max = self.find_grid_idx(np.array([x,y,height]))
                            for k in range(k_max):
                                obs_grid[k][i][j] = 1
            for coord in target_coords:
                x_idx, y_idx, z_idx = self.find_grid_idx(coord)
                obs_grid[z_idx][x_idx][y_idx] = 2
        else:
            for x_idx, y_idx, z_idx in obstacles:
                obs_grid[z_idx][x_idx][y_idx] = 1

            for x_idx, y_idx, z_idx in target_coords:
                obs_grid[z_idx][x_idx][y_idx] = 2

        return obs_grid

    def check_dist(self):
        build_centers = np.random.rand(self.num_builds, 2)
        for i in range(self.num_builds):
            for j in range(self.num_builds):
                while np.linalg.norm(build_centers[i] - build_centers[j]) < 1.414*self.width/self.dim and i!=j:
                    build_centers[i] = np.random.rand(1, 2)
        return build_centers

    def generate_random(self, min_height = 0.2):
        build_centers = self.check_dist()
        self.builds = []

        for i in range(self.num_builds):
            p_obj = Build()
            p_obj.centre = build_centers[i]
            index_x = math.floor(p_obj.centre[0] * self.dim / self.max_height)
            index_y = math.floor(p_obj.centre[1] * self.dim / self.max_height)
            p_obj.grid_index = [index_x, index_y]
            p_obj.top_right = [(index_x * self.max_height + 1.0) / self.dim, (index_y * self.max_height + 1) / self.dim, 0.0]
            p_obj.bottom_left =  [index_x * self.max_height / self.dim, index_y * self.max_height / self.dim, 0.0]
            p_obj.height = np.random.uniform(low = min_height, high = self.max_height)
            p_obj.poly2D_coords = ((p_obj.bottom_left[0], p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]+self.width/self.dim),
                            (p_obj.bottom_left[0], p_obj.bottom_left[1]+self.width/self.dim))
            self.builds.append(p_obj)

    def generate_grid(self):
        row_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        row_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        build_centers = []
        for i in range(len(row_x)):
            for j in range(len(row_y)):
                coord = [row_x[i], row_y[j]]
                build_centers.append(coord)
        build_centers = np.array(build_centers)

        self.builds = []

        for i in range(self.num_builds):
            p_obj = Build()
            p_obj.centre = build_centers[i]
            index_x = math.floor(p_obj.centre[0] * self.dim / self.max_height)
            index_y = math.floor(p_obj.centre[1] * self.dim / self.max_height)
            p_obj.grid_index = [index_x, index_y]
            p_obj.top_right = [(index_x * self.max_height + 1.0) / self.dim, (index_y * self.max_height + 1) / self.dim, 0.0]
            p_obj.bottom_left =  [index_x * self.max_height / self.dim, index_y * self.max_height / self.dim, 0.0]
            p_obj.height = np.random.uniform(low = 0.2, high = self.max_height)
            p_obj.poly2D_coords = ((p_obj.bottom_left[0], p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]+self.width/self.dim),
                            (p_obj.bottom_left[0], p_obj.bottom_left[1]+self.width/self.dim))
            self.builds.append(p_obj)





if __name__ == '__main__':
    trial = Obstacle(num_plants=10, test_name='grid')