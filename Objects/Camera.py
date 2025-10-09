
from pygame import Rect
import pygame
from Settings import global_settings as settings

class Camera:
    def __init__(self, x=0, y=0):
        
        self.window = pygame.display.set_mode((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))
        
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

    def reset_surface(self, surface):
        surface.fill((0, 0, 0, 0))

    def draw_surfaces(self, surfaces):
        
        # Draw the game world with the camera offset
        # for draw_rect in game_world.game_map.draw_rects:
            
        #     # Currenly this just draws all the rectangles in the game map
        #     # I will probably leave it like this so that later I can load in a level
        #     pygame.draw.rect(self.background_surface, draw_rect[1], 
        #                      (draw_rect[0].x - self.x, draw_rect[0].y - self.y, 
        #                       draw_rect[0].width, draw_rect[0].height))

        # Draw the player
        # player_one_rect = game_world.playerOne.hitbox.move(-self.x, -self.y) # move to camera position
        # pygame.draw.rect(self.player_surface, game_world.playerOne.color, player_one_rect)

        # player_two_rect = game_world.playerTwo.hitbox.move(-self.x, -self.y) # move to camera position
        # pygame.draw.rect(self.player_surface, game_world.playerTwo.color, player_two_rect)
        
        # Draw the camera surface to the window
        baseSurface = pygame.Surface((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT), pygame.SRCALPHA)
        baseSurface.set_alpha(None)
        for surface in surfaces:
            baseSurface.blit(self.background_surface, (0, 0))

        # Scale the merged surface and output it onto the window 
        self.window.blit(pygame.transform.scale(baseSurface, self.window.get_size()), (0, 0))

        for surface in surfaces:
            self.reset_surface(surface)

    # TODO: add a way to chage the camera size for a minimap
    # TODO: make a function to translate world coordinates to screen coordinates