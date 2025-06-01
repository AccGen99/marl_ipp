import copy
import os

import warnings
warnings.filterwarnings("ignore")

import imageio
import numpy as np
import torch
import time
import scipy.signal as signal
from multiprocessing import Pool

from test_env import Env
from attention_net import AttentionNet
from test_parameters import ENV_SIZE, NUM_AGENTS, EMBEDDING_DIM, INPUT_DIM, K_SIZE, FACING_ACTIONS

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class WorkerTest:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, device='cuda', greedy=False, save_image=False, seed=None):
        self.seed = seed
        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image

        self.env = Env(global_step, K_SIZE, NUM_AGENTS, budget_range, self.save_image, self.seed)
        self.local_net = localNetwork

    def run_episode(self, currEpisode):
        perf_metrics = dict()

        agents_list = self.env.reset() # Spawn agents in env
        self.sample_size = len(agents_list[0].action_coords)

        for agent in agents_list: # Initialize the agent networks
            agent.network = copy.deepcopy(self.local_net)
            agent.LSTM_h = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)
            agent.LSTM_c = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)
            agent.mask = torch.zeros((1, self.sample_size, K_SIZE*len(FACING_ACTIONS)), dtype=torch.int64).to(self.device)

        all_agents_functional = np.array([agent_obj.functional for agent_obj in agents_list])
        active_agent_ids = np.where(all_agents_functional == True)[0]

        for i in range(256):
            remain_budget = 0.0
            for active_agent in active_agent_ids:
                next_action = self.plan_next_action(agents_list[active_agent])
                _ = self.env.step_sample(agents_list[active_agent], next_action)
                remain_budget += agents_list[active_agent].budget

            self.env.test_post_processing(self.save_image)
            self.env.step += 1

            all_agents_functional = np.array([agent_obj.functional for agent_obj in agents_list])
            active_agent_ids = np.where(all_agents_functional == True)[0]

            print('{:.2f},{:.2f}'.format(remain_budget, 100*self.env.detected_targets))

            if remain_budget < 0.0 or len(active_agent_ids) == 0:
                remain_budget = 0
                for agent in agents_list:
                    remain_budget += agent.budget

                perf_metrics['detection_rate'] = 100*self.env.detected_targets

                print('{} Goodbye world! We did it!'.format(i))
                break

        print('Episode {} completed - {:.2f}%'.format(self.currEpisode, 100*self.env.detected_targets))

        return perf_metrics, agents_list

    def plan_next_action(self, agent):
        n_nodes = agent.action_coords.shape[0]
        node_util_inputs = agent.node_utils.reshape((n_nodes, 1))
        node_std_inputs = agent.node_std.reshape((n_nodes,1))
        node_neib_info = agent.neib_info.reshape((n_nodes,1))
        node_neib_std = agent.neib_std.reshape((n_nodes,1))
        budget_inputs = self.calc_estimate_budget(agent)
        node_inputs = np.concatenate((agent.action_coords, node_util_inputs, node_std_inputs, node_neib_info, node_neib_std), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 4)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 1)

        graph = list(agent.graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)

        current_index = torch.tensor([agent.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logp_list, value, agent.LSTM_h, agent.LSTM_c = agent.network(node_inputs, edge_inputs, budget_inputs, current_index, agent.LSTM_h, agent.LSTM_c, pos_encoding, agent.mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
        agent.route.append(next_node_index.item())
        agent.mask = torch.zeros((1, self.sample_size, K_SIZE*len(FACING_ACTIONS)), dtype=torch.int64).to(self.device)
        return next_node_index.item()

    def work(self, currEpisode, test):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        self.perf_metrics, self.all_agents = self.run_episode(currEpisode)

    def calc_estimate_budget(self, agent_obj):
        all_budget = []
        current_coord = agent_obj.action_coords[agent_obj.current_node_index]
        for i, point_coord in enumerate(agent_obj.action_coords):
            dist_current2point = agent_obj.controller.calcDistance(current_coord, point_coord)
            dist_point2end = 0
            estimate_budget = (agent_obj.budget - dist_current2point - dist_point2end) / 10
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i+1, 1)
    
    def calculate_position_embedding(self, edge_inputs):
        A_matrix = np.zeros((self.sample_size, self.sample_size))
        D_matrix = np.zeros((self.sample_size, self.sample_size))
        for i in range(self.sample_size):
            for j in range(self.sample_size):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.sample_size):
            D_matrix[i][i] = 1/np.sqrt(len(edge_inputs[i])-1)
        L = np.eye(self.sample_size) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:,1:32+1]
        return eigen_vector
    
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_det_{:.2f}.gif'.format(path, n, 100*self.env.detected_fruits), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)





if __name__=='__main__':
    device = torch.device('cuda')
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = WorkerTest(1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)
