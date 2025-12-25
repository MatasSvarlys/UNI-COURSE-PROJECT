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
MEMORY_SIZE = 100000
EXPERIENCE_COLLECTION_EPISODES = 25000

# every this amount of actions, optimize the policy network
NETWORK_LEARN_RATE = 10
# every this amount of policy network optimizations copy the policy network into the target network
NETWORK_SYNC_RATE = 1000

DISCOUNT_GAMA = 0.95
LEARNING_RATE = 0.0005

EPSILON_DECAY = 0.00000001
MIN_EPSILON = 0.1

FRAME_SKIPPING_STEPS = 4

REWARD_FOR_WINNING = 10
REWARD_FOR_EXISTING = 0.01
REWARD_FOR_PROXIMITY = 0
PENALTY_FOR_RUNNING_INTO_WALL = 0.0001
# TODO: make a penalty for not moving for a long period of time

START_REWARD = 0