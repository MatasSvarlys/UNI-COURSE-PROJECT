# True for agent control, False for human
RL_CONTROL = {
    "player_one": True,
    "player_two": False,
}

TRAINING_MODE = False
LOAD_MODEL = True

ACTIONS = [
    "NOOP",
    "LEFT",
    "RIGHT", 
    "JUMP",
    "LEFT_JUMP",
    "RIGHT_JUMP"
]

ACTION_SPACE_SIZE = len(ACTIONS)

MINI_BATCH = 128
MEMORY_SIZE = 200000
EXPERIENCE_COLLECTION_EPISODES = 5

# every this amount of actions, optimize the policy network
NETWORK_LEARN_RATE = 128
# every this amount of policy network optimizations copy the policy network into the target network
NETWORK_SYNC_RATE = 200

DISCOUNT_GAMA = 0.95
LEARNING_RATE = 0.0001

EPSILON_DECAY = 0.000001
MIN_EPSILON = 0.1

FRAMES_PER_STEP = 4
STEPS_PER_ACTION = 4

LIDAR_RAY_COUNT = 64
LIDAR_MAX_DISTANCE = 400

REWARD_FOR_WINNING = 10
REWARD_FOR_EXISTING = 0.001
REWARD_FOR_PROXIMITY = 0
PENALTY_FOR_RUNNING_INTO_WALL = 0.0001
# TODO: make a penalty for not moving for a long period of time

START_REWARD = 0

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84

CLASSIC_MODE = True
USE_DOUBLE_DQN = True