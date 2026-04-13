import json
import pygame
from Settings import global_settings as settings
from Settings import rl_settings

# I don't wanna rewrite the pygame name
vector = pygame.math.Vector2

class Player:
    # Public variables
    position: vector
    hitbox: pygame.Rect
    color: tuple
    movementVector: vector
    grounded: bool
    prev_direction: str
    isSeeker: bool

    def __init__(self, x, y, player_id, isSeeker=False):
        
        self.reward = 0

        self.position = vector(x, y)
        self.player_id = player_id
        self.isSeeker = isSeeker

        with open(f'./PlayerKeybinds/p{player_id+1}.json') as f:
            key_bindings = json.load(f)
        self.keymap = {action: getattr(pygame, code) for action, code in key_bindings.items()}


        self.hitbox = pygame.Rect(self.position.x, self.position.y, settings.PLAYER_WIDTH, settings.PLAYER_HEIGHT)
        
        if player_id == 0:
            self.color = settings.P1_PLAYER_COLOR
        else:
             self.color = settings.P2_PLAYER_COLOR
        
        self.movementVector = vector(0, 0)
        self.grounded = False
        self.prev_direction = "right"  # Track the last direction for sprite flipping and turning around while moving


        # self.isSeeker = not self.isSeeker

    # Update the player state
    def update(self, action, collisionRects, dt=1.0):
        
        # Movement vector from last frame
        last_movement = self.movementVector

        # Update player velocity from wasd/similar inputs
        input_vector = self.action_to_movement_vector(action, last_movement)

        # Apply gravity
        input_vector.y += settings.PLAYER_GRAVITY * dt

        # Handle max speed constraints and apply friction
        movement_vector = self.handle_constraints_and_friction(input_vector, dt)

        # If theres a wall in the way, move back to the edge of the wall
        movementVector, collisionDictionary = self.handle_collisions(collisionRects, movement_vector)
        
        self.movementVector = movementVector
        
        self.position.x = self.hitbox.x
        self.position.y = self.hitbox.y
        
        # This is the most literal way to do debuging every 60 frames
        # I probably should update this to do seconds instead
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1

        if settings.DEBUG_MODE and self._debug_counter % 60 == 0:
            print(f"Player {self.player_id} x velocity: {self.movementVector.x}")
            print(f"Player {self.player_id} y velocity: {self.movementVector.y}")
            print(f"Player {self.player_id} position: {self.position.x}, {self.position.y}")
            print(f"Player {self.player_id} hitbox: {self.hitbox.x}, {self.hitbox.y}")
            print(f"Player {self.player_id} grounded: {self.grounded}\n")
            print(f"Player {self.player_id} coliding: {collisionDictionary}")

    def handle_collisions(self, collisionRects, movementVector):
        
        collisionDictionary = {"left": False, "right": False, "up": False, "down": False}
        
        self.hitbox.move_ip(movementVector.x, 0)

        
        # ============== x collision ===============
        # get the position where the player would be if it moved

        # TODO: check only some surrounding blocks, checking every block slows down the game
        for rect in collisionRects:

            # Check for overlap and directly set your hitbox outside of the block
            if self.hitbox.colliderect(rect):
                
                self.reward -= rl_settings.PENALTY_FOR_RUNNING_INTO_WALL

                if movementVector.x > 0:  # right
                    # set the distance to move so the hitbox is exactly next to the rect
                    self.hitbox.right = rect.left
                    collisionDictionary["right"] = True
                elif movementVector.x < 0:  # left
                    self.hitbox.left = rect.right 
                    collisionDictionary["left"] = True
            
                # Reset your velocity, because you hit a wall 
                movementVector.x = 0

        # ================================

        # ============= Y collision ====================    

        self.hitbox.move_ip(0, movementVector.y)

        # Reset the check for grounded every frame, because if you're on the ground
        # and gravity is applied, you will collide and ground again.
        self.grounded = False

        # TODO: check only some surrounding blocks, checking every block slows down the game a lot
        for rect in collisionRects:

            if self.hitbox.colliderect(rect):
                if movementVector.y > 0:  # down
                    self.hitbox.bottom = rect.top 
                    self.grounded = True
                    collisionDictionary["down"] = True
                elif movementVector.y < 0:  # up
                    # Could be improved to give some downward velocity when bumping against a block
                    self.hitbox.top = rect.bottom
                    collisionDictionary["up"] = True

                movementVector.y = 0

        # ================================

        return movementVector, collisionDictionary
                    
    def handle_constraints_and_friction(self, movementVector, dt):
       
        # time_scale = dt * 60.0

        # Handle max fall speed
        if movementVector.y > settings.PLAYER_MAX_FSPEED:
            movementVector.y = settings.PLAYER_MAX_FSPEED
                
        # apply friction scaled by delta time
        # movementVector.x *= settings.PLAYER_FRICTION ** time_scale
        movementVector.x *= settings.PLAYER_FRICTION

        # zero out small movement
        if abs(movementVector.x) < 0.01:
            movementVector.x = 0

        # TODO: make the max speed not hard capped
        if movementVector.x > settings.PLAYER_MAX_SPEED:
            movementVector.x = settings.PLAYER_MAX_SPEED
        elif movementVector.x < -settings.PLAYER_MAX_SPEED:
            movementVector.x = -settings.PLAYER_MAX_SPEED
        
        return movementVector

    def action_to_movement_vector(self, action, lastMovement):
        # this is the theoretical movement vector for next frame, it will be processed later
        movementVector = vector(lastMovement.x, lastMovement.y)

        # --- x-axis movement ---

        # If direction keys are pressed, add acceleration to velocity each frame
        if rl_settings.ACTIONS[action] in ("LEFT", "LEFT_JUMP"):
            if self.prev_direction == "right":
                # I wonder if this has a name 
                # If the velocity is lower than the max flip acceleration, flip the velocity normally
                if lastMovement.x < settings.PLAYER_FLIP_MAX_VELOCITY:
                    movementVector.x = -lastMovement.x
                else:
                    # Otherwise, cap it to the max flip acceleration
                    movementVector.x = -settings.PLAYER_FLIP_MAX_VELOCITY
                self.prev_direction = "left"
            else:
                movementVector.x -= settings.PLAYER_ACCELERATION

        if rl_settings.ACTIONS[action] in ("RIGHT", "RIGHT_JUMP"):
            if self.prev_direction == "left":
            # If the velocity is higher than the negative max flip acceleration, flip the velocity normally
                if lastMovement.x > -settings.PLAYER_FLIP_MAX_VELOCITY:
                    movementVector.x = -lastMovement.x
                else:
                    # Otherwise, cap it to the max flip acceleration
                    movementVector.x = -settings.PLAYER_FLIP_MAX_VELOCITY
                self.prev_direction = "right"
            else:
                movementVector.x += settings.PLAYER_ACCELERATION


        # --- y-axis movement ---

        # If jump key is pressed and player is grounded, do the jump. 
        # We apply the base force here
        if rl_settings.ACTIONS[action] in ("JUMP", "LEFT_JUMP", "RIGHT_JUMP") and self.grounded:
            movementVector.y = -settings.PLAYER_JUMP_FORCE
            self.grounded = False

        # If we let go of jump while going up, the velocity gets cut hard, but not fully
        # if not key_inputs[self.keymap["JUMP"]] and lastMovement.y < 0:
        #     movementVector.y *= settings.PLAYER_JUMP_CUT_MULTIPLIER

        return movementVector
    
    def set_position(self, x, y):
        self.hitbox.x = x
        self.hitbox.y = y

        self.position.x = self.hitbox.x
        self.position.y = self.hitbox.y
        

    def draw_to_surface(self, surface): 
        pygame.draw.rect(surface, self.color, self.hitbox)