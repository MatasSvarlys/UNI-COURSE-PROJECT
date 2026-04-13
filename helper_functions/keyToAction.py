import json
from Settings import rl_settings
import pygame

_KEYBIND_CACHE = {}

def preload_keybinds(player_ids=[0, 1]):
    for pid in player_ids:
        try:
            with open(f'./PlayerKeybinds/p{pid+1}.json') as f:
                raw_bindings = json.load(f)
                # Convert string codes (like "K_LEFT") to pygame constants (like 1073741904)
                # We do this conversion here so we don't do it every frame!
                _KEYBIND_CACHE[pid] = {
                    action: getattr(pygame, code) 
                    for action, code in raw_bindings.items()
                }
        except FileNotFoundError:
            print(f"Warning: Keybind file for player {pid+1} not found.")

# Preload them immediately
preload_keybinds()
# ---------------------------------------------------------

def keys_to_action(player_id, keys):
    # Use the preloaded dictionary instead of opening a file
    keymap = _KEYBIND_CACHE.get(player_id)
    
    if not keymap:
        return rl_settings.ACTIONS.index("NOOP")

    # The logic remains the same, but it's now blazing fast
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