import json
import pygame
from Settings import global_settings as settings
from Objects.Map import Map

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
        
        self.position = vector(x, y)
        self.player_id = player_id
        self.isSeeker = isSeeker

        with open(f'./PlayerKeybinds/p{player_id}.json') as f:
            key_bindings = json.load(f)
        self.keymap = {action: getattr(pygame, code) for action, code in key_bindings.items()}


        self.hitbox = pygame.Rect(self.position.x, self.position.y, settings.PLAYER_WIDTH, settings.PLAYER_HEIGHT)
        
        self.color = settings.PLAYER_COLOR
        
        self.movementVector = vector(0, 0)
        self.grounded = False
        self.prev_direction = "right"  # Track the last direction for sprite flipping and turning around while moving

        self.coliding_y = False
        self.coliding_x = False

    # Update the player state
    def update(self, keyInputs, collisionRects):

        # Update player velocity from wasd/similar inputs
        input_vector = self.key_input_to_movement_vector(keyInputs)

        # Apply gravity and handle max speed constraints 
        constrained_vector = self.handle_constraints_gravity_and_friction(input_vector)

        # The previous two changed the velocity, now we clamp it further if theres a wall in the way
        final_vector = self.handle_collisions(collisionRects, constrained_vector)

        # Update position with final movement vector
        self.update_position(final_vector)

        # This is the most literal way to do debiging every 60 frames
        # I probably should update this to do seconds instead
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1

        if settings.DEBUG_MODE and self._debug_counter % 60 == 0:
            print(f"Player {self.player_id} x velocity: {self.velocity.x}")
            print(f"Player {self.player_id} y velocity: {self.velocity.y}")
            print(f"Player {self.player_id} position: {self.position_as_vector.x}, {self.position_as_vector.y}")
            print(f"Player {self.player_id} hitbox: {self.hitbox.x}, {self.hitbox.y}")
            print(f"Player {self.player_id} grounded: {self.grounded}\n")
            # I think due to the way collisions are handled, one frame I will be colliding and the next I'm forced to not
            # And that is why this flips between True and False kinda randomly 
            print(f"Player {self.player_id} coliding x: {self.coliding_x}")
            print(f"Player {self.player_id} coliding y: {self.grounded}\n")

    # for readability
    def update_position(self, movementVector):
        self.hitbox.move_ip(movementVector.x, movementVector.y)

    def handle_collisions(self, collisionRects, movementVector):
        
        self.coliding_x = False
        self.coliding_y = False


        # ============== x collision ===============
        # get the position where the player would be if it moved

        nextPos = self.hitbox.move(movementVector.x, 0)

        # TODO: check only some surrounding blocks, checking every block slows down the game
        for rect in collisionRects:

            # Check for overlap and directly set your hitbox outside of the block
            if nextPos.colliderect(rect):
                if movementVector.x > 0:  # right
                    # Calculate the distance to move so the hitbox is exactly next to the rect
                    movementVector.x = rect.left - self.hitbox.right
                elif movementVector.x < 0:  # left
                    movementVector.x = rect.right - self.hitbox.left
            
                # Reset your velocity, because you hit a wall 
                # and add debug flag 
                self.movementVector.x = 0
                self.coliding_x = True

        # ================================

        # ============= Y collision ====================
        # To save on calculations move the player hitbox directly
        self.hitbox.move(0, movementVector.y)
        
        # Reset the check for grounded every frame, because if you're on the ground
        # and gravity is applied, you will collide and ground again.
        self.grounded = False

        # TODO: check only some surrounding blocks, checking every block slows down the game a lot
        for rect in collisionRects:

            if self.hitbox.colliderect(rect):
                if movementVector.y > 0:  # down
                    movementVector.y = rect.top - self.hitbox.bottom
                    self.grounded = True
                elif movementVector.y < 0:  # up
                    movementVector.y = rect.bottom - self.hitbox.top
                
                # Reset velocity if we land or hit our head.
                # Could be improved to give some downward velocity when bumping against a block
                self.movementVector.y = 0
                self.coliding_y = True

        # ================================

        return movementVector
                    
                    

    def handle_constraints_gravity_and_friction(self, movementVector):
        
        self.movementVector += movementVector
        
        def approach_vel(curr, target=None, max=None, min=None):
            
            # If curr is out of provided bounds, slow it down 
            # because 0 < SLOWDOWN_FACTOR < 1
            if (max is not None and curr > max) or (min is not None and curr < min):
                return curr * settings.SLOWDOWN_FACTOR     
            else:
                if target:
                    return target
                else:
                    return curr 
        # =========================


        # IDK why, but approach_vel doesn't work for gravity
        self.movementVector.y += settings.PLAYER_GRAVITY
        if self.movementVector.y > settings.PLAYER_MAX_FSPEED:
            self.movementVector.y = settings.PLAYER_MAX_FSPEED

        # apply friction and stay within max speed
        self.movementVector.x = approach_vel(
            self.movementVector.x, 
            self.movementVector.x*settings.PLAYER_FRICTION, 
            settings.PLAYER_MAX_SPEED, 
            -settings.PLAYER_MAX_SPEED
        )
        
        if abs(self.movementVector.x) < 0.01:
            self.movementVector.x = approach_vel(self.movementVector.x, 0)
        
        # if the length of the movement vector still exeeds max speed normalize it to max speed  
        # yes it does make the approach_vel to max speed obsolete
        if self.movementVector.length() > settings.PLAYER_MAX_SPEED:
            self.movementVector = self.movementVector.normalize() * settings.PLAYER_MAX_SPEED

        return self.movementVector


    def key_input_to_movement_vector(self, key_inputs):
        # this is the theoretical movement vector for next frame, it will be processed later
        movementVector = vector(0, 0)

        # --- x-axis movement ---

        # If direction keys are pressed, add acceleration to velocity each frame
        if key_inputs[self.keymap["MOVE_LEFT"]]:
            if self.prev_direction == "right":
                # I wonder if this has a name 
                # If the velocity is lower than the max flip acceleration, flip the velocity normally
                if self.movementVector.x < settings.PLAYER_FLIP_MAX_VELOCITY:
                    movementVector.x = -self.movementVector.x
                else:
                    # Otherwise, cap it to the max flip acceleration
                    movementVector.x = -settings.PLAYER_FLIP_MAX_VELOCITY
                self.prev_direction = "left"
            else:
                movementVector.x -= settings.PLAYER_ACCELERATION

        if key_inputs[self.keymap["MOVE_RIGHT"]]:
            if self.prev_direction == "left":
            # If the velocity is higher than the negative max flip acceleration, flip the velocity normally
                if self.movementVector.x > -settings.PLAYER_FLIP_MAX_VELOCITY:
                    movementVector.x = -self.movementVector.x
                else:
                    # Otherwise, cap it to the max flip acceleration
                    movementVector.x = -settings.PLAYER_FLIP_MAX_VELOCITY
                self.prev_direction = "right"
            else:
                movementVector.x += settings.PLAYER_ACCELERATION


        # --- y-axis movement ---

        # If jump key is pressed and player is grounded, do the jump. 
        # We apply the base force here
        if key_inputs[self.keymap["JUMP"]] and self.grounded:
            movementVector.y -= settings.PLAYER_JUMP_FORCE
            self.grounded = False

        # If we let go of jump while going up, the velocity gets cut hard, but not fully
        if not key_inputs[self.keymap["JUMP"]] and self.velocity.y < 0:
            movementVector.y *= settings.PLAYER_JUMP_CUT_MULTIPLIER

        return movementVector
    
    def draw_to_surface(self, surface): 
        if self.isSeeker:
            pygame.draw.rect(surface, (255, 0, 0), self.hitbox)
        else:
            pygame.draw.rect(surface, self.color, self.hitbox)
        
        return surface
