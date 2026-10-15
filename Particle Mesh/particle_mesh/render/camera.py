import math

class SimulationCamera:
    def __init__(self, width, height, universe_w, universe_h):
        self.width = width
        self.height = height
        self.universe_w = universe_w
        self.universe_h = universe_h
        
        # Center camera in universe
        self.cam_x = (self.universe_w - self.width) / 2.0
        self.cam_y = (self.universe_h - self.height) / 2.0
        self.zoom = 1.0

    def reset(self):
        self.cam_x = (self.universe_w - self.width) / 2.0
        self.cam_y = (self.universe_h - self.height) / 2.0
        self.zoom = 1.0

    def pan(self, dx_screen, dy_screen):
        # Convert screen delta to world delta based on zoom
        self.cam_x -= dx_screen / self.zoom
        self.cam_y -= dy_screen / self.zoom

    def zoom_around_mouse(self, mouse_x_screen, mouse_y_screen, zoom_factor):
        # Find world coordinate under cursor before zoom
        world_x = (mouse_x_screen / self.zoom) + self.cam_x
        world_y = (mouse_y_screen / self.zoom) + self.cam_y
        
        # Apply zoom limit
        self.zoom = max(0.0001, min(10000.0, self.zoom * zoom_factor))
        
        # Re-offset camera to keep same world coordinate under cursor
        self.cam_x = world_x - (mouse_x_screen / self.zoom)
        self.cam_y = world_y - (mouse_y_screen / self.zoom)


    def screen_to_world(self, screen_x, screen_y):
        world_x = (screen_x / self.zoom) + self.cam_x
        world_y = (screen_y / self.zoom) + self.cam_y
        return world_x, world_y
