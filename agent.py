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
        self.route_actions = []

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