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
        self.handle_constraints_and_gravity()

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
            print(f"Player coliding y: {self.coliding_y}\n")


    def handle_collisions_and_update_position(self, game_map: Map):
        velocity_vec = self.velocity
        collision_rects = game_map.collision_rects

        # --- Handle X axis ---
        self.hitbox.move_ip(velocity_vec.x, 0)
        for rect in collision_rects:
            if self.hitbox.colliderect(rect):
                if velocity_vec.x > 0:  # moving right
                    self.hitbox.right = rect.left
                elif velocity_vec.x < 0:  # moving left
                    self.hitbox.left = rect.right
                self.velocity.x = 0
                self.coliding_x = True

        # --- Handle Y axis ---
        self.hitbox.move_ip(0, velocity_vec.y)
        self.grounded = False
        for rect in collision_rects:
            if self.hitbox.colliderect(rect):
                if velocity_vec.y > 0:  # moving down
                    self.hitbox.bottom = rect.top
                    self.grounded = True
                elif velocity_vec.y < 0:  # moving up
                    self.hitbox.top = rect.bottom
                self.velocity.y = 0

        # Update position as vector (after rect is final)
        self.position_as_vector = vector(self.hitbox.x, self.hitbox.y)

        print(self.position_as_vector)
                
                    
                    
                
                    

    def handle_constraints_and_gravity(self):

        # Apply gravity to vertical velocity
        # if not self.grounded:
        self.velocity.y += settings.PLAYER_GRAVITY


        # Clamp velocity to max speed
        self.velocity.x = max(-settings.PLAYER_MAX_SPEED, min(self.velocity.x, settings.PLAYER_MAX_SPEED))
        self.velocity.y = max(-settings.PLAYER_MAX_FSPEED, min(self.velocity.y, settings.PLAYER_MAX_FSPEED))
        
        # Trunc very small velocity values to zero to avoid floating point drift
        if abs(self.velocity.x) < 1e-6:
            self.velocity.x = 0

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

        # If no direction keys are pressed, apply friction to slow down
        if not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            self.velocity.x *= settings.PLAYER_FRICTION


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