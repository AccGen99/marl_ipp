BATCH_SIZE = 64 #1024
NUM_AGENTS = 3

INPUT_DIM = 8
EMBEDDING_DIM = 128
N_HEADS = 4
LR = 1e-4

FOLDER_NAME = 'our_approach'
DEPTH = 12
LOAD_MODEL = True
NUM_META_AGENT = 4

ENV_TYPE = 'grid'
K_SIZE = 20
BUDGET_RANGE = (4.0, 5.0) #(7.0, 9.0)
GEN_RANGE = [0.19, 0.21]
DIMS = 50
ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4
BETA = 1
LENGTH_SCALE = 0.5
COMMS_DIST = 0.3
FACING_ACTIONS = ['F', 'B', 'L', 'R'] # N, S, W, E

USE_GPU = False
USE_GPU_GLOBAL = True
CUDA_DEVICE = [0]

GAMMA = 1
EPSILON = 2e-1
DECAY_STEP = 512
SUMMARY_WINDOW = 1

model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
results_path = f'results/{FOLDER_NAME}'
logs_path = f'logs/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}/'
csv_path = results_path + '/csvs/'

SAVE_IMG_GAP = 1000