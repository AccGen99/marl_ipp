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

from env import Env
from attention_net import AttentionNet
from parameters import *

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_length, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length

        self.env = Env(global_step, K_SIZE, NUM_AGENTS, sample_length, budget_range, self.save_image)
        self.local_net = localNetwork
        self.experience = None

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
            all_agent_curr_coords = self.get_curr_coords(agents_list)
            p = Pool(processes=NUM_AGENTS)
            results = []
            for active_agent in active_agent_ids: # Execute only for agents who have budget > 0.0
                agent = agents_list[active_agent]
                # One process per active agent
                results.append(p.apply_async(self.plan_next_action, args=(agent,)))
            p.close()
            p.join()

            env_copy = copy.deepcopy(self.env)
            for res in results:
                env_copy_copy = copy.deepcopy(env_copy)
                updated_agent_obj, agent_id, action = res.get()
                agents_list[agent_id] = updated_agent_obj
                self.env.agents[agent_id] = copy.deepcopy(updated_agent_obj)
                reward, done = env_copy_copy.step_sample(self.env.agents[agent_id], action, save_img=False) # Get marginalized rewards for each agent
                self.env.agents[agent_id].ipp_reward = reward
                _ = self.env.step_sample(self.env.agents[agent_id], action, save_img=False) # Update the environment
            p.close()

            self.sample_size = len(agent.action_coords)
            self.env.post_processing(self.save_image)
            for agent in self.env.agents:
                total_reward = agent.ipp_reward + agent.comms_reward
                agent.experience[5] += torch.FloatTensor([[[total_reward]]]).to(self.device)
            self.env.step += 1

            all_agents_functional = np.array([agent_obj.functional for agent_obj in agents_list])
            active_agent_ids = np.where(all_agents_functional == True)[0]

            self.env.budget = 0.0
            for agent_id in active_agent_ids:
                agent = agents_list[agent_id]
                self.env.budget += agent.budget

            if self.env.budget < 0.0 or len(active_agent_ids) == 0:
                remain_budget = 0
                for agent in agents_list:
                    agent.experience[6] = agent.experience[4][1:]
                    agent.experience[6].append(torch.FloatTensor([[0]]).to(self.device))
                    remain_budget += agent.budget

                perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                perf_metrics['detection_rate'] = 100*self.env.detected_targets

                print('{} Goodbye world! We did it!'.format(i))
                break

        print('Episode {} completed - {:.2f}%'.format(self.currEpisode, 100*self.env.detected_targets))

        f = open(logs_path + '/train_log.txt', 'a')
        f.write('Episode {} completed with perf {:.2f}!\n'.format(self.currEpisode, 100*self.env.detected_targets))
        f.close()

        for agent in agents_list:
            reward = copy.deepcopy(agent.experience[5])
            reward.append(agent.experience[6][-1])
            reward_plus = np.array(reward,dtype=object).reshape(-1)
            discounted_rewards = discount(reward_plus, GAMMA)[:-1]
            discounted_rewards = discounted_rewards.tolist()
            target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)

            for i in range(target_v.size()[0]):
                agent.experience[7].append(target_v[i,:,:])

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, currEpisode)

        return perf_metrics, agents_list

    def plan_next_action(self, agent):
        torch.manual_seed(int(time.time())+100*agent.ID)
        agent.experience[9] += agent.LSTM_h
        agent.experience[10] += agent.LSTM_c
        agent.experience[11] += agent.mask

        n_nodes = agent.action_coords.shape[0]
        node_util_inputs = agent.node_utils.reshape((n_nodes, 1))
        node_std_inputs = agent.node_std.reshape((n_nodes,1))
        node_neib_info = agent.neib_info.reshape((n_nodes,1))
        node_neib_std = agent.neib_std.reshape((n_nodes,1))
        budget_inputs = self.calc_estimate_budget(agent)
        node_inputs = np.concatenate((agent.action_coords, node_util_inputs, node_std_inputs, node_neib_info, node_neib_std), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)

        graph = list(agent.graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)

        current_index = torch.tensor([agent.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,1)

        with torch.no_grad():
            logp_list, value, agent.LSTM_h, agent.LSTM_c = agent.network(node_inputs, edge_inputs, budget_inputs, current_index, agent.LSTM_h, agent.LSTM_c, pos_encoding, agent.mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        agent.value_list.append(value.squeeze(0).squeeze(0).item())

        agent.experience[0] += node_inputs
        agent.experience[1] += edge_inputs
        agent.experience[2] += current_index
        agent.experience[3] += action_index.unsqueeze(0).unsqueeze(0)
        agent.experience[4] += value
        agent.experience[8] += budget_inputs
        agent.experience[12] += pos_encoding

        next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
        agent.route.append(next_node_index.item())
        agent.mask = torch.zeros((1, self.sample_size, K_SIZE*len(FACING_ACTIONS)), dtype=torch.int64).to(self.device)

        return agent, agent.ID, next_node_index.item()

    def work(self, currEpisode):
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
    worker = Worker(1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)
