import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

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
    def __init__(self, ep_num, k_size, num_agents, budget_range, save_image=False):
        self.ep_num = ep_num
        self.num_agents = num_agents
        self.k_size = k_size
        self.budget = np.random.uniform(low = budget_range[0], high = budget_range[1])
        self.budget0 = deepcopy(self.budget)

        self.save_image = save_image
        self.frame_files = []

    def reset(self):
        self.obs = Obstacle(ENV_TYPE, self.num_agents, dim=DIMS)
        self.buildings = self.obs.get_buildings() # Generate buildings
        self.target_builds = Target_info(self.buildings, ENV_TYPE, self.num_agents, DIMS) # Generate windows on buildings
        self.all_target_coords = self.target_builds.get_all_coords() # Ground truth target distribution

        self.all_fruit_grids = []
        for coord in self.all_target_coords:
            i, j, k = self.obs.find_grid_idx(coord)
            self.all_fruit_grids.append([i, j, k])
        self.sensor = cam_sensor(self.all_fruit_grids, depth=DEPTH, shift=0.5, dim=DIMS)

        self.det_targets_cells = []
        self.detected_targets = 0.0
        self.total_targets = self.target_builds.num_targets
        self.utils_normalizer = 16.0
        self.env_gp = gp_3d(np.array([0.0, 0.0, 0.0, 0]), 4)
        env_high_info_area = self.env_gp.get_high_info_area() if ADAPTIVE_AREA else None
        self.cov_trace0 = self.env_gp.evaluate_cov_trace(env_high_info_area)

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
            agent_obj.neibs_gp.add_observed_point(np.array([0.0, 0.0, 0.0, 0]), 1.0)
            agent_obj.neibs_gp.update_gp()
            agent_obj.neib_info, agent_obj.neib_std = agent_obj.neibs_gp.gp.predict(agent_obj.action_coords, return_std=True)
            self.agents.append(agent_obj)
        
        self.step = 0
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
        utility, utils_rew, observed_targets, obstacles = self.sensor.get_utility_fast(grid_idx, facing)

        utility = utility/self.utils_normalizer
        self.detected_targets += utils_rew / self.total_targets
        obs_pt = np.array([sample[0], sample[1], sample[2], agent_obj.action_coords[int(next_node_index)][3]])

        self.env_gp.add_observed_point(obs_pt, utility)
        agent_obj.env_gp.add_observed_point(obs_pt, utility)

        # Get env uncertainty
        self.env_gp.update_gp()
        self.env_high_info_area = self.env_gp.get_high_info_area() if ADAPTIVE_AREA else None
        self.cov_trace = self.env_gp.evaluate_cov_trace(self.env_high_info_area)
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
        agent_obj.route_actions.append(agent_obj.action_coords[int(next_node_index)])

        try:
            assert self.budget >= 0  # Dijsktra filter
        except:
            done = True
            reward -= agent_obj.cov_trace / (DIMS*DIMS*DIMS*4)
        agent_obj.node_coords, agent_obj.action_coords, agent_obj.graph = agent_obj.controller.gen_graph(agent_obj.curr_coord[0:3], GEN_RANGE[0], GEN_RANGE[1])
        return reward, done

    def post_processing(self, save_image):
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
                            for sample in self.agents[j].route_actions:
                                obs_pt = np.array([sample[0], sample[1], sample[2], sample[3]])
                                if np.any(agent_obj.neibs_gp.observed_points != sample):
                                    agent_obj.neibs_gp.add_observed_point(obs_pt, 1.0)
                                if np.any(neib_obj.neibs_gp.observed_points != sample):
                                    neib_obj.neibs_gp.add_observed_point(obs_pt, 1.0)
                agent_obj.neibs_gp.update_gp()
                agent_obj.neib_info, agent_obj.neib_std = agent_obj.neibs_gp.gp.predict(agent_obj.action_coords, return_std=True)
                if agent_obj.comms_flag:
                    agent_obj.comms_flag = False
                    neib_high_info_area = agent_obj.neibs_gp.get_high_info_area() if ADAPTIVE_AREA else None
                    cov_trace = agent_obj.neibs_gp.evaluate_cov_trace(neib_high_info_area)
                    comms_reward = (agent_obj.prev_comms_trace - cov_trace) / agent_obj.prev_comms_trace
                    agent_obj.prev_comms_trace = cov_trace
                    agent_obj.comms_reward = comms_reward

        if save_image:
            self.visualize()

    def visualize(self):
        fig_s = (10,5.5)
        fig = plt.figure(figsize=fig_s)

        ax = fig.add_subplot(111, projection='3d')

        # Plot all targets as red circles
        for coord in self.all_target_coords:
            ax.plot(coord[0], coord[1], coord[2], '.', color='red')

        # Replace discovered targets red circles with green stars
        for grid_cell in self.sensor.already_observed:
            coords = self.obs.find_grid_coord(grid_cell)
            ax.plot(coords[0], coords[1], coords[2], '*', color='green')

        ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax.plot([0.0, 0.0], [0.0, 1.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax.plot([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        # Plot robot paths
        k = 0
        colors = ['red', 'blue', 'magenta', 'black']*16 # Repeat colors for upto 64 robots
        for agent in self.agents:
            for i in range(len(agent.route_coords)):
                if i+1 == len(agent.route) or len(agent.route) == 1 or len(agent.route) == 0:
                    break
                try:
                    x_vals = [agent.route_coords[i][0], agent.route_coords[i+1][0]]
                    y_vals = [agent.route_coords[i][1], agent.route_coords[i+1][1]]
                    z_vals = [agent.route_coords[i][2], agent.route_coords[i+1][2]]
                    ax.plot(x_vals, y_vals, z_vals, color=colors[k], linewidth=2.5)
                except:
                    pass

            ax.scatter(agent.route_coords[0][0], agent.route_coords[0][1], agent.route_coords[0][2], '.', color='darkgoldenrod', s=75.0)
            ax.scatter(agent.route_coords[-1][0], agent.route_coords[-1][1], agent.route_coords[-1][2], '.', color=colors[k], s=75.0)
            k += 1

        ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax.plot([0.0, 0.0], [0.0, 1.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax.plot([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        path = 'gifs/{}/{}/'.format(FOLDER_NAME, DIMS)
        if not os.path.exists(path):
            os.mkdir(path)

        path = path + '{}/'.format(self.num_agents)
        if not os.path.exists(path):
            os.mkdir(path)

        name = path + 'episode_{}_step_{}.png'.format(self.ep_num, self.step)

        remain_budget = 0.0
        if self.num_agents>1:
            for agent in self.agents:
                remain_budget += agent.budget
            
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