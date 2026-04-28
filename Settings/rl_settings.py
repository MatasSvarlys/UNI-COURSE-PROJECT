# True for agent control, False for human
RL_CONTROL = {
    "player_one": True,
    "player_two": True,
}

TRAINING_MODE = True
LOAD_MODEL = False

# How many steps for a log to be pushed. Only affects qvals and distributions logging
LOG_INTERVAL = 100

ACTIONS = [
    "NOOP",
    "LEFT",
    "RIGHT", 
    "JUMP",
    "LEFT_JUMP",
    "RIGHT_JUMP"
]

ACTION_SPACE_SIZE = len(ACTIONS)

# dependant on the gpu, but dont set it more than 256, since it can overlearn the non-catch memories
MINI_BATCH = 128
# Highly dependent on your ram, but at least 100k is recommended
MEMORY_SIZE = 200000
# how many experiences to collect before we start learning
MEMORY_COLLECTION_EPISODES = 100

SINGLE_AGENT_TRAINING_EPISODES = 20000
MAX_EPISODES = 200001

# every this amount of actions, optimize the policy network
NETWORK_LEARN_RATE = 128
# every this amount of policy network optimizations copy the policy network into the target network
NETWORK_SYNC_RATE = 200

DISCOUNT_GAMA = 0.95
LEARNING_RATE = 0.00005

EPSILON_DECAY = 0.99999
MIN_EPSILON = 0.1

FRAMES_PER_STEP = 4
STEPS_PER_ACTION = 4

LIDAR_RAY_COUNT = 32
LIDAR_MAX_DISTANCE = 400

REWARD_FOR_WINNING = 1
PENALTY_FOR_LOOSING = -0.5
REWARD_FOR_EXISTING = 0.05
PENALTY_FOR_EXISTING = -0.001
REWARD_FOR_PROXIMITY = 0
PENALTY_FOR_RUNNING_INTO_WALL = 0.0001
FRAME_REWARD_FOR_CATCHING = 50
# TODO: make a penalty for not moving for a long period of time

START_REWARD = 0

MAX_FRAMES = 1200

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84

# true - uses map screenshots, false - uses LIDAR observations
CLASSIC_MODE = False
# true - other player will always be visible
# false - direct LOS between players has to be made to appear on the map
TOGGLE_VISIBLE_PLAYERS_IN_OBSERVATION = True

USE_DOUBLE_DQN = True
USE_DUELING_DQN = True
USE_NOISY_NETS = True
USE_PRIORITIZED_EXPERIENCE_REPLAY = True

USE_DISTRIBUTIONAL_DQN = True
NUM_ATOMS = 51
VMAX = 2
VMIN = -2

N_STEP_LENGTH = 3