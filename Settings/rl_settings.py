# True for agent control, False for human
RL_CONTROL = {
    "player_one": True,
    "player_two": False,
}

ACTIONS = [
    "NOOP",
    "LEFT",
    "RIGHT", 
    "JUMP",
    "LEFT_JUMP",
    "RIGHT_JUMP"
]

ACTION_SPACE_SIZE = len(ACTIONS)

MINI_BATCH = 32
MEMORY_SIZE = 10000
NETWORK_SYNC_RATE = 20

DISCOUNT_GAMA = 0.99
LEARNING_RATE = 0.0001

EPSILON_DECAY = 0.001
MIN_EPSILON = 0.05
TRAINING_MODE = True