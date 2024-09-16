import numpy as np

from parameters import NUM_AGENTS



class Agent:
    def __init__(self, i):
        self.ID = i
        self.curr_coord = None
        self.prev_coord = None
        self.controller = None
        self.budget = None
        self.graph = None
        self.current_node_index = 0
        self.action_coords = None
        self.node_coords = None
        self.all_node_coords = None

        self.route = None
        self.route_coords = None

        self.gp = None
        self.node_utils = None
        self.node_std = None
        self.cov_trace = None

        self.network = None
        self.LSTM_h = None
        self.LSTM_c = None
        self.mask = None

        self.neibs_data = None
        self.neibs_coords = None
        self.reward = 0.0
        self.value_list = []
        self.ipp_reward = 0.0
        self.comms_reward = 0.0
        self.prev_comms_trace = 50*50*50*4
        self.curr_comms_trace = 50*50*50*4
        self.comms_flag = False
        self.functional = True

        self.terminated = False
        self.collided = False
        self.neib_info = None
        self.neib_std = None

        self.env_gp = None
        self.neibs_gp = None
        self.neib_inputs = None

        self.experience = []
        for i in range(14):
            self.experience.append([])

        # self.get_coords(500)
    
    # Generate body frame coords only once
    def get_coords(self, num_iters):
        self.body_coords = None
        for i in range(num_iters):
            x1 = np.random.normal(loc=0, scale=1)
            x2 = np.random.normal(loc=0, scale=1)
            x3 = np.random.normal(loc=0, scale=1)
            scale = 5*np.sqrt(x1*x1 + x2*x2 + x3*x3)
            if i == 0:
                self.body_coords = np.array([x1/scale, x2/scale, x3/scale]).reshape(-1,3)
            else:
                new_coord = np.array([x1/scale, x2/scale, x3/scale]).reshape(-1,3)
                self.body_coords = np.concatenate((self.body_coords, new_coord), axis=0)

    def generate_graph(self, new_coords, pre_process = False):
        self.node_coords, self.action_coords, self.graph = self.controller.generate_graph(new_coords, self.curr_coord, self.current_node_index)
        if NUM_AGENTS > 1:
            if pre_process:
                self.importance_weights = np.repeat(np.array([0.0]), len(self.action_coords))
            else:
                importance_weights = np.linalg.norm(self.node_coords - self.nearest_neib, axis = 1)
                self.importance_weights = np.repeat(importance_weights, 4)