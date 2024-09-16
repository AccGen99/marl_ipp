INPUT_DIM = 8
EMBEDDING_DIM = 128
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 1
NUM_META_AGENT = 1
GAMMA = 1
FOLDER_NAME = 'run_1'
model_path = f'model/{FOLDER_NAME}'
result_path = f'result/{FOLDER_NAME}'

SEED = 1
NUM_TEST = 25
SAVE_IMG_GAP = 1

BUDGET_RANGE = (9.99999, 10)
K_SIZE = 20
SAMPLE_LENGTH = 0.2

TEST_TYPE = 'random'
FACING_ACTIONS = ['F', 'B', 'L', 'R']
logs_path = f'result/{FOLDER_NAME}'
csv_path = f'CSVs/{TEST_TYPE}'
