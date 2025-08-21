
from pygame import Rect
import pygame
from Settings import global_settings as settings

class Camera:
    def __init__(self, size, x=0, y=0):
        self.surface = pygame.Surface(size)

        self.x = x
        self.y = y

    # ============ The basics ============
    def manual_nudge(self, dx, dy):
        self.x += dx
        self.y += dy

    def follow(self, target: Rect) -> None:
        self.x = target.x - settings.SCREEN_WIDTH // 2
        self.y = target.y - settings.SCREEN_HEIGHT // 2
    # ====================================

    def follow_with_offset(self, target: Rect, offset_x=0, offset_y=0) -> None:

        # TODO: add clamping to the size of the map
        self.x = target.x - settings.SCREEN_WIDTH // 2 + offset_x
        self.y = target.y - settings.SCREEN_HEIGHT // 2 + offset_y

    def get_position(self):
        return self.x, self.y
  
    def reset(self):
        self.x = 0
        self.y = 0

    def draw_world(self, game_world, window):
        
        self.surface.fill((0, 0, 0))

        # Draw the game world with the camera offset
        for draw_rect in game_world.game_map.draw_rects:
            
            # Currenly this just draws all the rectangles in the game map
            # I will probably leave it like this so that later I can load in a level
            pygame.draw.rect(self.surface, draw_rect[1], 
                             (draw_rect[0].x - self.x, draw_rect[0].y - self.y, 
                              draw_rect[0].width, draw_rect[0].height))

        # Draw the player
        player_rect = game_world.player.hitbox.move(-self.x, -self.y)
        pygame.draw.rect(self.surface, game_world.player.color, player_rect)

        # Draw the camera surface to the window
        scaled_canvas = pygame.transform.scale(self.surface, window.get_size())
        window.blit(scaled_canvas, (0, 0))

    # TODO: add a way to chage the camera size for a minimap