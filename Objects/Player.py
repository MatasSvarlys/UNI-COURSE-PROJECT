import pygame
from Settings import global_settings as settings
from Objects.Map import Map

# I don't wanna rewrite the pygame name
vector = pygame.math.Vector2

class Player:
    # Public variables
    position_as_vector: vector
    hitbox: pygame.Rect
    color: tuple
    velocity: vector
    grounded: bool
    prev_direction: str

    def __init__(self, x, y, game_map):
        
        self.position_as_vector = vector(x, y)
        self.map = game_map

        self.hitbox = pygame.Rect(self.position_as_vector.x, self.position_as_vector.y, settings.PLAYER_WIDTH, settings.PLAYER_HEIGHT)
        
        self.color = settings.PLAYER_COLOR
        
        self.velocity = vector(0, 0)
        self.grounded = False
        self.prev_direction = "right"  # Track the last direction for sprite flipping

        self.coliding_y = False
        self.coliding_x = False

    # Update the player state
    def update(self, keys, game_map: Map):

        # Update player velocity from wasd/similar inputs
        self.handle_player_movement(keys)

        # Handle max speed, gravity, velocity normalization and truncation
        self.handle_constraints_gravity_and_friction()

        # The previous two changed the velocity, now we clamp it further if theres a wall in the way
        self.handle_collisions_and_update_position(game_map)


        # This is the most literal way to do debiging every 60 frames
        # I probably should update this to do seconds instead
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1

        if settings.DEBUG_MODE and self._debug_counter % 60 == 0:
            print(f"Player x velocity: {self.velocity.x}")
            print(f"Player y velocity: {self.velocity.y}")
            print(f"Player position: {self.position_as_vector.x}, {self.position_as_vector.y}")
            print(f"Player hitbox: {self.hitbox.x}, {self.hitbox.y}")
            print(f"Player grounded: {self.grounded}\n")
            print(f"Player coliding x: {self.coliding_x}")
            print(f"Player coliding y: {self.grounded}\n")


    def handle_collisions_and_update_position(self, game_map: Map):
        
        velocity_vec = self.velocity
        collision_rects = game_map.collision_rects

        self.coliding_x = False
        self.coliding_y = False


        # ============== x collision ===============
        # To save on calculations move the player hitbox directly
        self.hitbox.move_ip(velocity_vec.x, 0)

        # TODO: check only some surrounding blocks, checking every block slows down the game a lot
        for rect in collision_rects:

            # Check for overlap and directly set your hitbox outside of the block
            if self.hitbox.colliderect(rect):
                if velocity_vec.x > 0:  # right
                    self.hitbox.right = rect.left
                elif velocity_vec.x < 0:  # left
                    self.hitbox.left = rect.right
            
                # Reset your velocity, because you hit a wall 
                # and add debug flag 
                self.velocity.x = 0
                self.coliding_x = True


        # ============= Y collision ====================
        # To save on calculations move the player hitbox directly
        self.hitbox.move_ip(0, velocity_vec.y)
        
        # Reset the check for grounded every frame, because if you're on the ground
        # and gravity is applied, you will collide and ground again.
        self.grounded = False

        # TODO: check only some surrounding blocks, checking every block slows down the game a lot
        for rect in collision_rects:

            # y collisions 
            if self.hitbox.colliderect(rect):
                if velocity_vec.y > 0:  # down
                    self.hitbox.bottom = rect.top
                    self.grounded = True
                elif velocity_vec.y < 0:  # up
                    self.hitbox.top = rect.bottom
                
                # Reset velocity if we land or hit our head.
                # Could be improved to give some downward velocity when bumping against a block
                self.velocity.y = 0
                self.coliding_y = True

        # ================================

        # Update position as vector (after hitbox is final)
        self.position_as_vector = vector(self.hitbox.x, self.hitbox.y)        
                    
                    

    def handle_constraints_gravity_and_friction(self):
        
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

        # Yes, this doesn't cap max jump speed
        self.velocity.y = approach_vel(
            self.velocity.y,
            self.velocity.y + settings.PLAYER_GRAVITY,
            settings.PLAYER_MAX_FSPEED
        )

        self.velocity.x = approach_vel(self.velocity.x, self.velocity.x*settings.PLAYER_FRICTION, settings.PLAYER_MAX_SPEED, -settings.PLAYER_MAX_SPEED)
        
        if abs(self.velocity.x) < 0.01:
            self.velocity.x = approach_vel(self.velocity.x, 0)
        



        # ======== Legacy code ========
        # # Apply gravity to vertical velocity
        # # if not self.grounded:
        # self.velocity.y += settings.PLAYER_GRAVITY


        # # Clamp velocity to max speed
        # self.velocity.x = max(-settings.PLAYER_MAX_SPEED, min(self.velocity.x, settings.PLAYER_MAX_SPEED))
        # self.velocity.y = max(-settings.PLAYER_MAX_FSPEED, min(self.velocity.y, settings.PLAYER_MAX_FSPEED))
        
        # # Trunc very small velocity values to zero to avoid floating point drift
        # if abs(self.velocity.x) < 1e-6:
        #     self.velocity.x = 0
        # ========================

        # Before the position, normalize the velocity vector to ensure consistent speed
        
        if self.velocity.length() > 0:
            self.velocity = self.velocity.normalize() * self.velocity.length()


    def handle_player_movement(self, keys):

        # --- x-axis movement ---

        # If direction keys are pressed, add acceleration to velocity each frame
        # TODO: make the keys customizable
        if keys[pygame.K_LEFT]:
            if self.prev_direction == "right":
                # I wonder if this has a name 
            # If the velocity is lower than the max flip acceleration, flip the velocity normally
                if self.velocity.x < settings.PLAYER_FLIP_MAX_VELOCITY:
                    self.velocity.x = -self.velocity.x
                else:
                    # Otherwise, cap it to the max flip acceleration
                    self.velocity.x = -settings.PLAYER_FLIP_MAX_VELOCITY
                self.prev_direction = "left"
            else:
                self.velocity.x -= settings.PLAYER_ACCELERATION

        if keys[pygame.K_RIGHT]:
            if self.prev_direction == "left":
            # If the velocity is higher than the negative max flip acceleration, flip the velocity normally
                if self.velocity.x > -settings.PLAYER_FLIP_MAX_VELOCITY:
                    self.velocity.x = -self.velocity.x
                else:
                    # Otherwise, cap it to the max flip acceleration
                    self.velocity.x = -settings.PLAYER_FLIP_MAX_VELOCITY
                self.prev_direction = "right"
            else:
                self.velocity.x += settings.PLAYER_ACCELERATION

        # --- y-axis movement ---

        # If jump key is pressed and player is grounded, do the jump. 
        # We apply the base force here
        if keys[pygame.K_UP] and self.grounded:
            self.velocity.y -= settings.PLAYER_JUMP_FORCE
            self.grounded = False

        # If we let go of jump while going up, the velocity gets cut hard, but not fully
        if not keys[pygame.K_UP] and self.velocity.y < 0:
            self.velocity.y *= settings.PLAYER_JUMP_CUT_MULTIPLIER







    # =========== Legacy code ===========
    # def draw(self, screen, camera):
    #     camera_x, camera_y = camera.get_position()
    #     # Adjust player's position relative to the camera
    #     draw_rect = self.hitbox.move(-camera_x, -camera_y)
        
        
    #     if not hasattr(self, "_debug_counter_draw"):
    #         self._debug_counter_draw = 0
    #     self._debug_counter_draw += 1
    #     if settings.DEBUG_MODE and self._debug_counter_draw % 60 == 0:
    #         print(f"Drawing player at: {draw_rect.x}, {draw_rect.y}\n")
        
    #     pygame.draw.rect(screen, self.color, draw_rect)
    # ====================================