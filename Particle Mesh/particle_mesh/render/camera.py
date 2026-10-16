import math

# Zoom-in limit: one world unit spans this many screen pixels
MAX_ZOOM = 10000.0


class SimulationCamera:
    def __init__(self, width, height, universe_w, universe_h):
        self.width = width
        self.height = height
        self.universe_w = universe_w
        self.universe_h = universe_h
        # Zoom-out limit: the view never spans more than the physics mesh along either axis
        self.min_zoom = min(MAX_ZOOM, max(width / universe_w, height / universe_h))
        self.max_zoom = MAX_ZOOM
        self.reset()

    def reset(self):
        """Centre the view on the universe at zoom 1 (or the zoom-out limit, if the mesh is smaller than the
        window)."""
        self.zoom = max(1.0, self.min_zoom)
        self.cam_x = (self.universe_w - self.width / self.zoom) / 2.0
        self.cam_y = (self.universe_h - self.height / self.zoom) / 2.0

    def pan(self, dx_screen, dy_screen):
        # Convert screen delta to world delta based on zoom
        self.cam_x -= dx_screen / self.zoom
        self.cam_y -= dy_screen / self.zoom

    def zoom_around_mouse(self, mouse_x_screen, mouse_y_screen, zoom_factor):
        """Scale the zoom by zoom_factor, clamped to [min_zoom, max_zoom], keeping the world point under the cursor
        fixed. Non-finite or non-positive factors are ignored."""
        if not (math.isfinite(zoom_factor) and zoom_factor > 0.0):
            return
        # Find world coordinate under cursor before zoom
        world_x = (mouse_x_screen / self.zoom) + self.cam_x
        world_y = (mouse_y_screen / self.zoom) + self.cam_y

        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * zoom_factor))

        # Re-offset camera to keep same world coordinate under cursor
        self.cam_x = world_x - (mouse_x_screen / self.zoom)
        self.cam_y = world_y - (mouse_y_screen / self.zoom)

    def screen_to_world(self, screen_x, screen_y):
        world_x = (screen_x / self.zoom) + self.cam_x
        world_y = (screen_y / self.zoom) + self.cam_y
        return world_x, world_y
