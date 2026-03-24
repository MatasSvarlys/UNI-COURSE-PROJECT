import json
from Settings import rl_settings
import pygame


def keys_to_action(player_id, keys):
    with open(f'./PlayerKeybinds/p{player_id+1}.json') as f:
        key_bindings = json.load(f)

    keymap = {action: getattr(pygame, code) for action, code in key_bindings.items()}


    if keys[keymap["MOVE_LEFT"]] and keys[keymap["JUMP"]]:
        return rl_settings.ACTIONS.index("LEFT_JUMP")

    if keys[keymap["MOVE_RIGHT"]] and keys[keymap["JUMP"]]:
        return rl_settings.ACTIONS.index("RIGHT_JUMP")

    if keys[keymap["MOVE_LEFT"]]:
        return rl_settings.ACTIONS.index("LEFT")

    if keys[keymap["MOVE_RIGHT"]]:
        return rl_settings.ACTIONS.index("RIGHT")

    if keys[keymap["JUMP"]]:
        return rl_settings.ACTIONS.index("JUMP")

    return rl_settings.ACTIONS.index("NOOP")
