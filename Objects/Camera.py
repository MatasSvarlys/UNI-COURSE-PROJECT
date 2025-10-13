
from pygame import Rect
import pygame
from Settings import global_settings as settings

class Camera:
    def __init__(self, x=0, y=0):
        
        self.window = pygame.display.set_mode((settings.DISPLAY_WIDTH, settings.DISPLAY_HEIGHT))

        
        self.x = x
        self.y = y

    # ============ The basics ============
    def manual_nudge(self, dx, dy):
        self.x += dx
        self.y += dy

    def follow(self, target: Rect) -> None:
        self.x = target.x - settings.WINDOW_WIDTH // 2
        self.y = target.y - settings.WINDOW_HEIGHT // 2
    # ====================================

    def follow_with_offset(self, target: Rect, offset_x=0, offset_y=0) -> None:

        # TODO: add clamping to the size of the map
        self.x = target.x - settings.WINDOW_WIDTH // 2 + offset_x
        self.y = target.y - settings.WINDOW_HEIGHT // 2 + offset_y

    def follow_between_players(self, player1: Rect, player2: Rect) -> None:
        # Calculate the midpoint between the two players
        mid_x = (player1.centerx + player2.centerx) // 2
        mid_y = (player1.centery + player2.centery) // 2
        # Center the camera on the midpoint
        self.x = mid_x - settings.WINDOW_WIDTH // 2
        self.y = mid_y - settings.WINDOW_HEIGHT // 2

    def get_position(self):
        return self.x, self.y
  
    def reset(self):
        self.x = 0
        self.y = 0

    def reset_surface(self, surface):
        surface.fill((0, 0, 0, 0))

    def draw_surfaces(self, surfaces):
        self.window.fill((0, 0, 0))
        
        baseSurface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        baseSurface.fill((0, 0, 0))

        # Merge all surfaces
        for surface in surfaces:
            baseSurface.blit(surface, (0, 0))

        # Scale the merged surface and output it onto the window 
        self.window.blit(pygame.transform.scale(baseSurface, self.window.get_size()), (self.x + settings.WINDOW_WIDTH // 2, self.y + settings.WINDOW_HEIGHT // 2))
        pygame.display.flip()

        # reset all surfaces for next frame
        for surface in surfaces:
            self.reset_surface(surface)

    # TODO: add a way to chage the camera size for a minimap
    # TODO: make a function to translate world coordinates to screen coordinates