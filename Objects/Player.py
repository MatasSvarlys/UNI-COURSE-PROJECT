import pygame
from Settings import global_settings as settings

vector = pygame.math.Vector2

class Player:
    def __init__(self, x, y):
        self.position = vector(x, y)
        self.hitbox = pygame.Rect(self.position.x, self.position.y, settings.PLAYER_WIDTH, settings.PLAYER_HEIGHT)
        self.color = settings.PLAYER_COLOR
        self.velocity = vector(0, 0)
        self.grounded = False
        self.prev_direction = "right"  # Track the last direction for sprite flipping



    def update(self, keys):

        # If direction keys are pressed, add acceleration to velocity each frame
        # TODO: make the keys customizable
        if keys[pygame.K_LEFT]:
            if self.prev_direction == "right":
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

        # If jump key is pressed and player is grounded, do the jump. 
        # We apply the base force here
        if keys[pygame.K_UP] and self.grounded:
            self.velocity.y -= settings.PLAYER_JUMP_FORCE
            self.grounded = False

        # If we let go of jump while going up, the velocity gets cut hard, but not fully
        if not keys[pygame.K_UP] and self.velocity.y < 0:
            self.velocity.y *= settings.PLAYER_JUMP_CUT_MULTIPLIER

        # If no direction keys are pressed, apply friction to slow down
        if not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            self.velocity.x *= settings.PLAYER_FRICTION



        # Clip very small velocity values to zero to avoid floating point drift
        if abs(self.velocity.x) < 1e-6:
            self.velocity.x = 0

        # Apply gravity to vertical velocity
        self.velocity.y += settings.PLAYER_GRAVITY


        # Apply acceleration to velocity and clamp speed
        if self.velocity.x > settings.PLAYER_MAX_SPEED:
            self.velocity.x = settings.PLAYER_MAX_SPEED
        elif self.velocity.x < -settings.PLAYER_MAX_SPEED:
            self.velocity.x = -settings.PLAYER_MAX_SPEED

        if self.grounded:
            self.velocity.y = 0


        # Before the position, normalize the velocity vector to ensure consistent speed
        if self.velocity.length() > 0:
            self.velocity = self.velocity.normalize() * self.velocity.length()

        # Update position
        self.position += self.velocity
        self.hitbox.x = self.position.x
        self.hitbox.y = self.position.y


        # If the player is on the ground, reset vertical velocity
        # TODO: check for collisions with the ground instead of just checking the position
        if self.position.y >= settings.SCREEN_HEIGHT - settings.PLAYER_HEIGHT and not self.grounded:
            self.grounded = True
            self.position.y = settings.SCREEN_HEIGHT - settings.PLAYER_HEIGHT



        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1
        if settings.DEBUG_MODE and self._debug_counter % 60 == 0:
            print(f"Player x velocity: {self.velocity.x}")
            print(f"Player y velocity: {self.velocity.y}")
            print(f"Player position: {self.position.x}, {self.position.y}")
            print(f"Player hitbox: {self.hitbox.x}, {self.hitbox.y}")
            print(f"Player grounded: {self.grounded}\n")






    def draw(self, screen, camera):
        camera_x, camera_y = camera.get_position()
        # Adjust player's position relative to the camera
        draw_rect = self.hitbox.move(-camera_x, -camera_y)
        
        
        if not hasattr(self, "_debug_counter_draw"):
            self._debug_counter_draw = 0
        self._debug_counter_draw += 1
        if settings.DEBUG_MODE and self._debug_counter_draw % 60 == 0:
            print(f"Drawing player at: {draw_rect.x}, {draw_rect.y}\n")
        
        pygame.draw.rect(screen, self.color, draw_rect)
