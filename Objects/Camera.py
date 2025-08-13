from Settings import global_settings as settings

class Camera:
    def __init__(self, 
                 display_size_x=settings.SCREEN_WIDTH, 
                 display_size_y=settings.SCREEN_HEIGHT, 
                 x=0, 
                 y=0):
        self.display_size_x = display_size_x
        self.display_size_y = display_size_y
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def get_position(self):
        return self.x, self.y
  
    def reset(self):
        self.x = 0
        self.y = 0

    # TODO: add a way to chnage the camera size after initialization