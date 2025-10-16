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