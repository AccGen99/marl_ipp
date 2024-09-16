import copy
import csv
import os
import ray
import torch
import time
from multiprocessing import Pool
import numpy as np
import time
from attention_net import AttentionNet
from runner import Runner
from test_worker import WorkerTest
from test_parameters import *


def run_test():
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    global_network.load_state_dict(checkpoint['model'])

    print(f'Loading model: {FOLDER_NAME}...')
    print(f'Total budget range: {BUDGET_RANGE}')

    # init meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.to(local_device).state_dict() if device != local_device else global_network.state_dict()
    curr_test = 1
    metric_name = ['detection_rate']

    try:
        while True:
            jobList = []
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(meta_agent.job.remote(weights, curr_test, budget_range=BUDGET_RANGE, sample_length=SAMPLE_LENGTH))
                curr_test += 1
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)
            # print('Working 1')
            for job in done_jobs:
                metrics, info = job

            if curr_test > NUM_TEST:
                break


    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=8/NUM_META_AGENT, num_gpus=NUM_GPU/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)

    def singleThreadedJob(self, episodeNumber, budget_range, sample_length):
        save_img = True if episodeNumber % SAVE_IMG_GAP == 0 else False
        seed = SEED + 100 * episodeNumber
        np.random.seed(seed)
        worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=seed)
        worker.work(episodeNumber, 0)
        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, global_weights, episodeNumber, budget_range, sample_length=None):
        self.set_weights(global_weights)
        metrics = self.singleThreadedJob(episodeNumber, budget_range, sample_length)

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(1):
        run_test()
