import pygame
import global_settings as settings


class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, settings.PLAYER_WIDTH, settings.PLAYER_HEIGHT)
        self.color = settings.PLAYER_COLOR
        self.speed = settings.PLAYER_SPEED
        
    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
