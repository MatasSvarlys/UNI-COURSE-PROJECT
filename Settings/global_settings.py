
# Screen
from Settings import map_settings

#TODO: make screen size non-constant

WINDOW_WIDTH = map_settings.MAP_WIDTH * map_settings.TILE_SIZE
WINDOW_HEIGHT = map_settings.MAP_HEIGHT * map_settings.TILE_SIZE
DISPLAY_WIDTH = WINDOW_WIDTH * 2
DISPLAY_HEIGHT = WINDOW_HEIGHT * 2

# Color constants (TODO: make this to sprites down the line)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACKGROUND_COLOR = (30, 30, 30)

# Player settings
PLAYER_COLOR = (255, 0, 0)
PLAYER_WIDTH = 20
PLAYER_HEIGHT = 20
PLAYER_MAX_SPEED = 50
PLAYER_MAX_FSPEED = 15
PLAYER_ACCELERATION = 1.5
PLAYER_FRICTION = 0.8
PLAYER_GRAVITY = 0.5
PLAYER_JUMP_FORCE = 10
PLAYER_FLIP_MAX_VELOCITY = 2 
PLAYER_JUMP_CUT_MULTIPLIER = 0.5  
SLOWDOWN_FACTOR = 0.3

# Tiles are square
# This should not be changed lightly
TILE_SIZE_RAW = 64

# Camera settings
CAMERA_MODE = False
DEBUG_MODE = False


# Gen collision consts
PRECISION_MULTIPLYER = 2

HEADLESS_MODE = False