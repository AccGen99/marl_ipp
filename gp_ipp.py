import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
# from classes.Gaussian2D import Gaussian2D
from parameters import *


class gp_3d():
    def __init__(self, node_coords, s):
        self.shape = s
        if self.shape == 4:
            self.kernel = Matern(length_scale=0.5, nu=0.1)
        else:
            self.kernel = Matern(length_scale=0.2)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=None, n_restarts_optimizer=0)
        self.observed_points = []
        self.observed_value = []
        self.node_coords = node_coords

    def add_observed_point(self, point_pos, value):
        self.observed_points.append(point_pos)
        self.observed_value.append(value)

    def update_gp(self):
        if self.observed_points:
            X = np.array(self.observed_points).reshape(-1,self.shape)
            y = np.array(self.observed_value).reshape(-1,1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y)

    def update_node(self):
        y_pred, std = self.gp.predict(self.node_coords, return_std=True)

        return y_pred, std

    def evaluate_RMSE(self, y_true):
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)
        x3 = np.linspace(0, 1, 50)
        x4 = np.linspace(0, 3, 4)
        if self.shape == 4:
            x1x2 = np.array(list(product(x1, x2, x3, x4)))
        else:
            x1x2 = np.array(list(product(x1, x2, x3)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        RMSE = np.sqrt(mean_squared_error(y_pred, y_true))
        return RMSE

    def evaluate_F1score(self, y_true):
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)
        x3 = np.linspace(0, 1, 50)
        x4 = np.linspace(0, 3, 4)
        if self.shape == 4:
            x1x2x3 = np.array(list(product(x1, x2, x3, x4)))
        else:
            x1x2x3 = np.array(list(product(x1, x2, x3)))
        score = self.gp.score(x1x2x3,y_true)
        return score

    def evaluate_cov_trace(self, X=None):
        if X is None:
            x1 = np.linspace(0, 1, 50)
            x2 = np.linspace(0, 1, 50)
            x3 = np.linspace(0, 1, 50)
            x4 = np.linspace(0, 3, 4)
        # x1x2 = np.array(list(product(x1, x2, x3, x4)))
            if self.shape == 4:
                X = np.array(list(product(x1, x2, x3, x4)))
            else:
                X = np.array(list(product(x1, x2, x3)))
        _, std = self.gp.predict(X, return_std=True)
        trace = np.sum(std*std)
        return trace

    def return_predictions(self):
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)
        x3 = np.linspace(0, 1, 50)
        x4 = np.linspace(0, 3, 4)
        if self.shape == 4:
            X = np.array(list(product(x1, x2, x3, x4)))
        else:
            X = np.array(list(product(x1, x2, x3)))
        X = np.array(list(product(x1, x2, x3, x4)))
        info, _ = self.gp.predict(X, return_std=True)
        return info

    def evaluate_mutual_info(self, X=None): ################################################################################
        if X is None:
            x1 = np.linspace(0, 1, 50)
            x2 = np.linspace(0, 1, 50)
            x3 = np.linspace(0, 1, 50)
            x4 = np.linspace(0, 3, 4)
            if self.shape == 4:
                X = np.array(list(product(x1, x2, x3, x4)))
            else:
                X = np.array(list(product(x1, x2, x3)))
        n_sample = X.shape[0]
        _, cov = self.gp.predict(X, return_cov=True)
        
        mi = (1 / 2) * np.log(np.linalg.det(0.01*cov.reshape(n_sample, n_sample) + np.identity(n_sample)))
        return mi

    def get_high_info_area(self, t=ADAPTIVE_TH, beta=BETA):
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)
        x3 = np.linspace(0, 1, 50)
        x4 = np.linspace(0, 3, 4)
        if self.shape == 4:
            x1x2 = np.array(list(product(x1, x2, x3, x4)))
        else:
            x1x2 = np.array(list(product(x1, x2, x3)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        
        high_measurement_area = []
        if self.shape == 4:
            val = 50 * 50 * 50 * 4
        else:
            val = 50 * 50 * 50
        interest_arr = []
        # interest = np.zeros((val, 1))
        interest = y_pred.reshape(-1,1) + beta * std.reshape(-1,1)
        truth_arr = interest >= t
        truth_arr = truth_arr.reshape(-1)
        # truth_arr = np.repeat(truth_arr, 4)
        # print(x1x2.shape) # (500000, 1)
        # print(x1x2[0])
        high_area = np.array(x1x2[truth_arr])
        # print('high_area - {}'.format(high_area))

        # for i in range(val):
        #     interest = y_pred[i] + beta * std[i]
        #     interest_arr.append(interest)
        #     if y_pred[i] + beta * std[i] >= t:
        #         high_measurement_area.append(x1x2[i])
        # high_measurement_area = np.array(high_measurement_area)
        # print('interest_arr_max - {}'.format(max(interest_arr)))
        return high_area #high_measurement_area





if __name__ == '__main__':
    pass