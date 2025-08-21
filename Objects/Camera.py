
from pygame import Rect
import pygame
from Settings import global_settings as settings

class Camera:
    def __init__(self, size, x=0, y=0):
        self.surface = pygame.display.set_mode((size[0], size[1]))

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
        self.x = target.x - settings.SCREEN_WIDTH // 2 + offset_x
        self.y = target.y - settings.SCREEN_HEIGHT // 2 + offset_y

    def get_position(self):
        return self.x, self.y
  
    def reset(self):
        self.x = 0
        self.y = 0

    def draw_world(self, game_world):
        # Draw the game world with the camera offset
        for draw_rect in game_world.game_map.draw_rects:
            
            # Adjust the rect position based on the camera position
            # TODO: update the way I store draw_rects so that I can just pass a rect to draw
            pygame.draw.rect(self.surface, draw_rect[1], 
                             (draw_rect[0].x, draw_rect[0].y, 
                              draw_rect[0].width, draw_rect[0].height))

        # Draw the player
        player_rect = game_world.player.hitbox
        pygame.draw.rect(self.surface, game_world.player.color, player_rect)

    def render_to_window(self, window, window_x=0, window_y=0):
        window.blit(self.surface, (window_x, window_y))
    # TODO: add a way to chage the camera size for a minimap