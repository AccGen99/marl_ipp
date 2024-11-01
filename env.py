import numpy as np
import sklearn.metrics as metrics
from scipy.stats import entropy
from matplotlib import pyplot as plt
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

from classes.Target import Target_info
from classes.Builds import Obstacle
from classes.Controller import obsController
from classes.Sensor import cam_sensor
from agent import Agent

from gp_ipp import gp_3d
from parameters import *



class Env():
    def __init__(self, ep_num, k_size, num_agents, sample_length, budget_range, save_image=False, seed=None):
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

        self.ep_num = ep_num
        self.num_builds = 25
        self.sample_length = sample_length
        self.num_agents = num_agents

        self.obs = Obstacle(self.num_builds, ENV_TYPE)
        self.buildings = self.obs.get_buildings()      # Generate buildings
        self.target_builds = Target_info(self.buildings, ENV_TYPE) # Generate windows on buildings
        self.all_target_coords = self.target_builds.get_all_coords()      # Ground truth

        self.k_size = k_size
        self.sensor = cam_sensor(depth=DEPTH, shift=0.5)
        self.budget = np.random.uniform(low = budget_range[0], high = budget_range[1])
        self.budget0 = deepcopy(self.budget)

        self.env_RMSE = None
        self.env_F1score = None
        self.env_cov_trace = None
        self.env_MI = None
        self.env_entropy = None
        
        # start point
        self.save_image = save_image
        self.frame_files = []

    def reset(self, seed=None):
        # underlying distribution
        self.total_budget_arr = []
        self.total_detected_arr = []
        self.obs = Obstacle(self.num_builds, ENV_TYPE)
        self.buildings = self.obs.get_buildings()
        self.target_builds = Target_info(self.buildings, ENV_TYPE)
        self.detected_targets = 0.0
        self.all_target_coords = self.target_builds.get_all_coords()      # Ground truth
        self.ground_truth = self.obs.get_3d_gt(self.all_target_coords)
        self.gt_occupancy_grid = self.obs.get_gt_occupancy_grid(self.all_target_coords, is_ground_truth=True)
        self.occupancy_grid = np.zeros((DIMS, DIMS, DIMS))

        self.total_targets = self.target_builds.num_targets
        self.env_detected_targets = 0

        # initialize evaluations
        self.env_gp = gp_3d(np.array([0.0, 0.0, 0.0, 0]), 4)
        self.RMSE = metrics.mean_squared_error(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten(), squared=False)
        self.F1score = metrics.f1_score(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten(), average='weighted')
        self.MI = metrics.mutual_info_score(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten())
        self.high_info_area = self.env_gp.get_high_info_area() if ADAPTIVE_AREA else None
        self.cov_trace = self.env_gp.evaluate_cov_trace(self.high_info_area)
        self.entropy = entropy(self.occupancy_grid.flatten(), self.gt_occupancy_grid.flatten())
        self.cov_trace0 = deepcopy(self.cov_trace)
        self.utils_normalizer = 16.0

        # Initialize agents
        self.agents = []
        for i in range(self.num_agents):
            agent_obj = Agent(i)
            agent_obj.curr_coord = np.array([0.0, 0.0, 0.0, 0])
            agent_obj.current_node_index = 0
            agent_obj.route = [agent_obj.current_node_index]
            agent_obj.route_coords = [agent_obj.curr_coord]
            agent_obj.budget = deepcopy(self.budget) / self.num_agents
            agent_obj.controller = obsController(agent_obj.curr_coord[0:3], self.k_size)
            agent_obj.node_coords, agent_obj.action_coords, agent_obj.graph = agent_obj.controller.gen_graph(agent_obj.curr_coord[0:3], GEN_RANGE[0], GEN_RANGE[1])
            agent_obj.node_utils = np.zeros((len(agent_obj.action_coords),1))
            agent_obj.node_std = np.ones((len(agent_obj.action_coords),1))
            agent_obj.cov_trace = DIMS*DIMS*DIMS*4
            agent_obj.env_gp = gp_3d(np.array([0.0, 0.0, 0.0, 0]), 4)
            agent_obj.neibs_gp = gp_3d(np.array([0.0, 0.0, 0.0, 0]), 4)
            agent_obj.neibs_gp.add_observed_point(np.array([0.0, 0.0, 0.0, 0]), 0.0)
            agent_obj.neibs_gp.update_gp()
            agent_obj.neib_info, agent_obj.neib_std = agent_obj.neibs_gp.gp.predict(agent_obj.action_coords, return_std=True)
            self.agents.append(agent_obj)
        
        self.curr_step_vals = []
        self.curr_step_samples = []

        for i in range(self.num_agents):
            neibs_coords = []
            for j in range(self.num_agents):
                if i!=j:
                    neibs_coords += list(self.agents[j].curr_coord[0:3])
            self.agents[i].neibs_coords = neibs_coords

        self.step = 0
        self.closest_dist_to_neib = 0.0
        return self.agents

    def step_sample(self, agent_obj, next_node_index):
        dist = np.linalg.norm(agent_obj.curr_coord[0:3] - agent_obj.action_coords[next_node_index][0:3])
        agent_obj.prev_coord = agent_obj.curr_coord[0:3]
        facing = agent_obj.action_coords[int(next_node_index)][3]
        facing = FACING_ACTIONS[int(facing)]

        utils_rew = 0
        done = False # Budget not exhausted

        sample = agent_obj.action_coords[next_node_index][0:3]
        grid_idx = self.obs.find_grid_idx(sample)
        utility, utils_rew, observed_targets, obstacles = self.sensor.get_utility(grid_idx, self.gt_occupancy_grid, facing)

        self.occupancy_grid = self.obs.get_gt_occupancy_grid(observed_targets, obstacles, prev_grid=self.occupancy_grid)
        utility = utility/self.utils_normalizer
        self.detected_targets += utils_rew / self.total_targets
        obs_pt = np.array([sample[0], sample[1], sample[2], agent_obj.action_coords[int(next_node_index)][3]])

        self.env_gp.add_observed_point(obs_pt, utility)
        agent_obj.env_gp.add_observed_point(obs_pt, utility)

        # Get env uncertainty
        self.env_gp.update_gp()
        agent_obj.env_gp.update_gp()
        env_high_info_area = agent_obj.env_gp.get_high_info_area() if ADAPTIVE_AREA else None
        cov_trace = agent_obj.env_gp.evaluate_cov_trace(env_high_info_area)

        # REWARD
        reward = 0
        exp_reward = 0.0
        if next_node_index in agent_obj.route[-1:]: # if revisiting
            reward += -0.01
        if agent_obj.cov_trace > cov_trace: # if reducing uncertainty
            exp_reward = (agent_obj.cov_trace - cov_trace) / agent_obj.cov_trace
        agent_obj.cov_trace = cov_trace
        reward += utils_rew / 50.0 + 20.0*exp_reward

        if (int(agent_obj.current_node_index) - int(next_node_index)) // 4 == 0:
            dist += 0.05
        agent_obj.budget -= dist

        agent_obj.current_node_index = next_node_index
        agent_obj.curr_coord = agent_obj.action_coords[next_node_index]
        agent_obj.route.append(int(next_node_index))
        agent_obj.route_coords.append(agent_obj.action_coords[int(next_node_index)][0:3])

        try:
            assert self.budget >= 0  # Dijsktra filter
        except:
            done = True
            reward -= agent_obj.cov_trace / (DIMS*DIMS*DIMS*4)
        agent_obj.node_coords, agent_obj.action_coords, agent_obj.graph = agent_obj.controller.gen_graph(agent_obj.curr_coord[0:3], GEN_RANGE[0], GEN_RANGE[1])
        return reward, done

    def post_processing(self, save_image):
        for i in range(self.num_agents):
            neibs_coords = []
            for j in range(self.num_agents):
                if i!=j:
                    neibs_coords += list(self.agents[j].curr_coord[0:3])
            self.agents[i].neibs_coords = neibs_coords

        self.min_dist = np.float16('inf')
        for i in range(self.num_agents):
            agent = self.agents[i]
            best_dist = np.float16('inf')
            for j in range(self.num_agents):
                if i != j:
                    neib = self.agents[j]
                    dist = np.linalg.norm(neib.curr_coord[0:3] - agent.curr_coord[0:3])
                    if dist < best_dist:
                        agent.nearest_neib = deepcopy(neib.curr_coord)
                        best_dist = dist
                    if dist < self.min_dist:
                        self.min_dist = dist

        for i in range(self.num_agents):
            if self.agents[i].functional == True:
                agent_obj = self.agents[i]
                agent_obj.env_gp = deepcopy(self.env_gp)
                agent_obj.node_utils, agent_obj.node_std = agent_obj.env_gp.gp.predict(agent_obj.action_coords, return_std=True)
                for j in range(i, self.num_agents):
                    neib_obj = self.agents[j]
                    if self.agents[j].functional == True:
                        if i != j:
                            dist = np.linalg.norm(agent_obj.curr_coord[0:3] - neib_obj.curr_coord[0:3])
                            if dist < COMMS_DIST:
                                agent_obj.comms_flag = True
                                neib_obj.comms_flag = True
                            # Global comms during training
                            sample = self.agents[j].curr_coord
                            obs_pt = np.array([sample[0], sample[1], sample[2], sample[3]])
                            agent_obj.neibs_gp.add_observed_point(obs_pt, 0.0)
                            neib_obj.neibs_gp.add_observed_point(obs_pt, 0.0)
                agent_obj.neibs_gp.update_gp()
                agent_obj.neib_info, agent_obj.neib_std = agent_obj.neibs_gp.gp.predict(agent_obj.action_coords, return_std=True)
                if agent_obj.comms_flag:
                    agent_obj.comms_flag = False
                    neib_high_info_area = agent_obj.neibs_gp.get_high_info_area() if ADAPTIVE_AREA else None
                    cov_trace = agent_obj.neibs_gp.evaluate_cov_trace(neib_high_info_area)
                    comms_reward = (agent_obj.prev_comms_trace - cov_trace) / agent_obj.prev_comms_trace
                    agent_obj.prev_comms_trace = cov_trace
                    agent_obj.comms_reward = 0.0 #comms_reward

        if save_image:
            self.visualize()
        return self.min_dist

    def test_post_processing(self, save_image):
        for i in range(self.num_agents):
            neibs_coords = []
            for j in range(self.num_agents):
                if i!=j:
                    neibs_coords += list(self.agents[j].curr_coord[0:3])
            self.agents[i].neibs_coords = neibs_coords

        self.min_dist = np.float16('inf')
        for i in range(self.num_agents):
            agent = self.agents[i]
            best_dist = np.float16('inf')
            for j in range(self.num_agents):
                if i != j:
                    neib = self.agents[j]
                    dist = np.linalg.norm(neib.curr_coord[0:3] - agent.curr_coord[0:3])
                    if dist < best_dist:
                        agent.nearest_neib = deepcopy(neib.curr_coord)
                        best_dist = dist
                    if dist < self.min_dist:
                        self.min_dist = dist

        for i in range(self.num_agents):
            if self.agents[i].functional == True:
                agent_obj = self.agents[i]
                for j in range(i, self.num_agents):
                    neib_obj = self.agents[j]
                    if self.agents[j].functional == True:
                        if i != j:
                            dist = np.linalg.norm(agent_obj.curr_coord[0:3] - neib_obj.curr_coord[0:3])
                            if dist < COMMS_DIST:
                                self.num_comms += 1
                                agent_obj.comms_flag = True
                                neib_obj.comms_flag = True
                                # Share env_gp information
                                for i in range(len(neib_obj.obs_vals)):
                                    agent_obj.new_obs_coords.append(neib_obj.obs_coords[i])
                                    agent_obj.new_obs_vals.append(neib_obj.obs_vals[i])
                                for i in range(len(agent_obj.obs_vals)):
                                    neib_obj.new_obs_coords.append(agent_obj.obs_coords[i])
                                    neib_obj.new_obs_vals.append(agent_obj.obs_vals[i])
                                # Share current location information
                                sample = self.agents[j].curr_coord
                                obs_pt = np.array([sample[0], sample[1], sample[2], sample[3]])
                                agent_obj.neibs_gp.add_observed_point(obs_pt, 0.0)
                                neib_obj.neibs_gp.add_observed_point(obs_pt, 0.0)
                agent_obj.neibs_gp.update_gp()
                agent_obj.neib_info, agent_obj.neib_std = agent_obj.neibs_gp.gp.predict(agent_obj.action_coords, return_std=True)
                if agent_obj.comms_flag:
                    agent_obj.comms_flag = False
                    neib_high_info_area = agent_obj.neibs_gp.get_high_info_area() if ADAPTIVE_AREA else None
                    cov_trace = agent_obj.neibs_gp.evaluate_cov_trace(neib_high_info_area)
                    comms_reward = (agent_obj.prev_comms_trace - cov_trace) / agent_obj.prev_comms_trace
                    agent_obj.prev_comms_trace = cov_trace
                    agent_obj.comms_reward = 0.0 #comms_reward

        for i in range(self.num_agents):
            agent_obj = self.agents[i]
            agent_obj.update_env_gp()
            agent_obj.node_utils, agent_obj.node_std = agent_obj.env_gp.gp.predict(agent_obj.action_coords, return_std=True)

        if save_image:
            self.visualize()

        return self.min_dist

    def visualize(self):
        fig_s = (10,5.5)
        fig = plt.figure(figsize=fig_s)

        # GROUND TRUTH PLOTTING
        l = 1.0 / self.obs.dim
        ax = fig.add_subplot(121, projection='3d', label='Ground Truth')
        lw = 0.25

        for each_build in self.buildings: #self.target_builds:
            bl = each_build.bottom_left
            tr = [bl[0] + self.obs.width*l, bl[1] + self.obs.width*l]
            h = each_build.height

            ax.plot( [bl[0], bl[0]], [bl[1], bl[1]], [0.0, h], color='blue', linewidth=lw)
            ax.plot([bl[0] + self.obs.width*l, bl[0] + self.obs.width*l], [bl[1], bl[1]], [0.0, h], color='blue', linewidth=lw)
            ax.plot([tr[0], tr[0]], [tr[1], tr[1]], [0.0, h], color='blue', linewidth=lw)
            ax.plot([bl[0], bl[0]], [bl[1]+self.obs.width*l, bl[1]+self.obs.width*l], [0.0, h], color='blue', linewidth=lw)

            ax.plot([bl[0], tr[0], bl[0]+self.obs.width*l, bl[0], bl[0]], [bl[1]+self.obs.width*l, tr[1], bl[1], bl[1], bl[1]+self.obs.width*l], [h, h, h, h, h], color='blue', linewidth=lw)#, c='k', marker=None, linestyle = '-', linewidth = lw)
            ax.plot([bl[0], tr[0], bl[0]+self.obs.width*l, bl[0], bl[0]], [bl[1]+self.obs.width*l, tr[1], bl[1], bl[1], bl[1]+self.obs.width*l], [0.0, 0.0, 0.0, 0.0, 0.0], color='blue')#, c='k', marker=None, linestyle = '-', linewidth = lw)

        ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        # Plotting fruits
        target_x = []
        target_y = []
        target_z = []
        for coord in self.all_target_coords:
            target_x.append(coord[0])
            target_y.append(coord[1])
            target_z.append(coord[2])
        ax.plot(target_x, target_y, target_z, '*', color='green')

        # ROBOT BELIEF PLOTTING
        ax1 = fig.add_subplot(122, projection='3d')

        for coord in self.all_target_coords:
            coord = self.obs.find_grid_coord(self.obs.find_grid_idx(coord))
            ax1.plot(coord[0], coord[1], coord[2], '.', color='red')

        for k in range(self.obs.dim):
            for j in range(self.obs.dim):
                for i in range(self.obs.dim):
                    grid_cell = [i, j, k]
                    coords = self.obs.find_grid_coord(grid_cell)
                    val = self.occupancy_grid[k][i][j]
                    if val == 1:
                        ax1.plot(coords[0], coords[1], coords[2], '|', color='black', alpha=0.2)
                    if val == 2:
                        ax1.plot(coords[0], coords[1], coords[2], '*', color='green')
        ax1.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 0.0], [0.0, 1.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        # ROBOT ROUTE, GRAPH AND SENSOR VIEW AREA PLOTTING
        k = 0
        colors = ['red', 'blue', 'magenta', 'black']
        for agent in self.agents:
            for i in range(len(agent.route_coords)):
                if i+1 == len(agent.route) or len(agent.route) == 1 or len(agent.route) == 0:
                    break
                try:
                    x_vals = [agent.route_coords[i][0], agent.route_coords[i+1][0]]
                    y_vals = [agent.route_coords[i][1], agent.route_coords[i+1][1]]
                    z_vals = [agent.route_coords[i][2], agent.route_coords[i+1][2]]
                    ax1.plot(x_vals, y_vals, z_vals, color=colors[k], linewidth=2.5)
                except:
                    pass

            ax1.scatter(agent.route_coords[0][0], agent.route_coords[0][1], agent.route_coords[0][2], '.', color='darkgoldenrod', s=75.0)
            ax1.scatter(agent.route_coords[-1][0], agent.route_coords[-1][1], agent.route_coords[-1][2], '.', color=colors[k], s=75.0)
            k += 1

        ax1.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 0.0], [0.0, 1.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        name = gifs_path + 'episode_{}_step_{}.png'.format(self.ep_num, self.step)

        remain_budget = 0.0
        if self.num_agents>1:
            coll = False
            self.closest_dist_to_neib = np.float16('inf')
            for agent in self.agents:
                remain_budget += agent.budget
                dist = np.linalg.norm(agent.curr_coord[0:3] - agent.nearest_neib[0:3])
                if dist < self.closest_dist_to_neib:
                    self.closest_dist_to_neib = dist
            
            plt.suptitle('Budget: {:.2f}/{:.2f}, Det: {:.2f}%'.format(remain_budget, self.budget0, 100*self.detected_targets))
        else:
            plt.suptitle('Step: {}, Detected: {:.4g}%'.format(self.step, 100*self.detected_targets))

        plt.tight_layout()
        plt.savefig(name)
        self.frame_files.append(name)
        plt.close('all')





if __name__=='__main__':
    trial = Env(20)
    trial.visualize()