
from pygame import Rect
import pygame
from Settings import global_settings as settings

class Camera:
    def __init__(self, x=0, y=0):
        
        self.window = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))
        
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

        # Merge all surfaces
        baseSurface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        baseSurface.set_alpha(None)
        for surface in surfaces:
            baseSurface.blit(surface, (0, 0))

        # Scale the merged surface and output it onto the window 
        self.window.blit(pygame.transform.scale(baseSurface, self.window.get_size()), (0, 0))
        pygame.display.flip()

        # reset all surfaces for next frame
        for surface in surfaces:
            self.reset_surface(surface)

    # TODO: add a way to chage the camera size for a minimap
    # TODO: make a function to translate world coordinates to screen coordinates