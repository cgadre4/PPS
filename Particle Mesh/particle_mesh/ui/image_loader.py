import os
import numpy as np

def browse_custom_image():
    """
    Opens a file dialog to browse for a custom astro-image.
    Returns the absolute file path and the base filename, or (None, None).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        file_path = filedialog.askopenfilename(
            title="Select Astro-Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")]
        )
        root.destroy()
        
        if file_path:
            return file_path, os.path.basename(file_path)
    except Exception as e:
        print(f"Error browsing image: {e}")
    return None, None

def load_image_particles(file_path, cam_x, cam_y, zoom, spawn_density, width, height, view_w, view_h, render_cell_size, universe_w, universe_h):
    """
    Loads an image, resizes it to fit the screen aspect ratio,
    and stochastically generates particle coordinates based on pixel grayscale intensity.
    
    Returns a numpy array of shape (N, 2) with coordinates.
    """
    try:
        from PIL import Image, ImageOps
        
        img_raw = Image.open(file_path)
        img_oriented = ImageOps.exif_transpose(img_raw)
        img_rgb = img_oriented.convert('RGB')
        
        w_raw, h_raw = img_rgb.size
        
        aspect_img = float(w_raw) / float(h_raw)
        aspect_screen = float(view_w) / float(view_h)
        
        if aspect_img > aspect_screen:
            target_w = view_w
            target_h = int(view_w / aspect_img)
        else:
            target_h = view_h
            target_w = int(view_h * aspect_img)
            
        target_w = max(1, target_w)
        target_h = max(1, target_h)
        
        img_resized = img_rgb.resize((target_w, target_h), Image.Resampling.BICUBIC)
        img_data = np.array(img_resized, dtype=np.float32)
        intensities = np.mean(img_data, axis=2)
        
        col_start = (view_w - target_w) // 2
        row_start = (view_h - target_h) // 2
        
        cell_w = (width / zoom) / float(view_w)
        cell_h = (height / zoom) / float(view_h)
        
        cols_mesh, rows_mesh = np.meshgrid(np.arange(target_w), np.arange(target_h))
        screen_cols_all = col_start + cols_mesh
        screen_rows_all = row_start + rows_mesh
        
        x_world_all = cam_x + (screen_cols_all + 0.5) * cell_w
        y_world_all = cam_y + (view_h - screen_rows_all - 0.5) * cell_h
        
        in_grid_mask = (x_world_all >= 0.0) & (x_world_all < float(universe_w)) & \
                       (y_world_all >= 0.0) & (y_world_all < float(universe_h))
        
        expected_counts = (intensities / 255.0) * spawn_density * (render_cell_size ** 2)
        expected_counts[~in_grid_mask] = 0.0
        
        floor_counts = np.floor(expected_counts).astype(np.int32)
        fractional_parts = expected_counts - floor_counts
        rand_vals = np.random.rand(*expected_counts.shape)
        cell_counts = floor_counts + (rand_vals < fractional_parts).astype(np.int32)
        
        r_indices, c_indices = np.where(cell_counts > 0)
        counts_to_spawn = cell_counts[r_indices, c_indices]
        
        total_spawn = int(np.sum(counts_to_spawn))
        if total_spawn == 0:
            print(f"[image_loader] '{os.path.basename(file_path)}' produced no particles "
                  "(too dark, outside the universe, or spawn density too low)")
            return None
            
        screen_cols = col_start + np.repeat(c_indices, counts_to_spawn)
        screen_rows = row_start + np.repeat(r_indices, counts_to_spawn)
        
        new_pos = np.zeros((total_spawn, 2), dtype=np.float32)
        new_pos[:, 0] = cam_x + (screen_cols + np.random.rand(total_spawn)) * cell_w
        new_pos[:, 1] = cam_y + (view_h - screen_rows - 1 + np.random.rand(total_spawn)) * cell_h
        
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, float(universe_w - 0.001))
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, float(universe_h - 0.001))
        
        return new_pos
    except Exception as e:
        print(f"Error stochastically generating particles from image: {e}")
    return None
