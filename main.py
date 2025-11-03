import numpy as np
import math
from PIL import Image
import os
from nanba import raytrace_kernel

SagA_rs = 2.0 

WIDTH = 800
HEIGHT = 600
COMPUTE_WIDTH = 200
COMPUTE_HEIGHT = 150

cam_pos = np.array([0.0, -2.0, 20.0]) 
cam_target = np.array([0.0, 0.0, 0.0])
fov_y_deg = 60.0

disk_r1 = SagA_rs * 2
disk_r2 = SagA_rs * 5
disk_opacity = 0.9 

BACKGROUND_IMAGE_DATA = np.zeros((1, 1, 3), dtype=np.uint8) 
DISK_IMAGE_DATA = np.zeros((1, 1, 3), dtype=np.uint8)
BACKGROUND_WIDTH = 1
BACKGROUND_HEIGHT = 1
DISK_WIDTH = 1
DISK_HEIGHT = 1

if __name__ == "__main__":
    
    try:
        bg_img_pil = Image.open("image5.jpg").convert("RGB")
        BACKGROUND_IMAGE_DATA = np.array(bg_img_pil, dtype=np.uint8)
        BACKGROUND_HEIGHT, BACKGROUND_WIDTH, _ = BACKGROUND_IMAGE_DATA.shape

        disk_img_pil = Image.open("image8.jpg").convert("RGB")
        DISK_IMAGE_DATA = np.array(disk_img_pil, dtype=np.uint8)
        DISK_HEIGHT, DISK_WIDTH, _ = DISK_IMAGE_DATA.shape
        
        print("Successfully loaded image files.")

    except FileNotFoundError as e:
        print(f"Warning: {e}. Using procedural fallback images.")
        
        BACKGROUND_WIDTH = 100
        BACKGROUND_HEIGHT = 100
        BACKGROUND_IMAGE_DATA = np.zeros((BACKGROUND_HEIGHT, BACKGROUND_WIDTH, 3), dtype=np.uint8)
        
        DISK_WIDTH = 256
        DISK_HEIGHT = 256
        DISK_IMAGE_DATA = np.zeros((DISK_HEIGHT, DISK_WIDTH, 3), dtype=np.uint8)
        
        yellow = np.array([255.0, 255.0, 0.0])
        orange = np.array([255.0, 165.0, 0.0])
        
        r_grad = np.linspace(yellow[0], orange[0], DISK_HEIGHT, dtype=np.uint8)
        g_grad = np.linspace(yellow[1], orange[1], DISK_HEIGHT, dtype=np.uint8)
        b_grad = np.linspace(yellow[2], orange[2], DISK_HEIGHT, dtype=np.uint8)
        
        DISK_IMAGE_DATA[:, :, 0] = r_grad[:, np.newaxis]
        DISK_IMAGE_DATA[:, :, 1] = g_grad[:, np.newaxis]
        DISK_IMAGE_DATA[:, :, 2] = b_grad[:, np.newaxis]

    fwd = cam_target - cam_pos
    fwd_norm = np.linalg.norm(fwd)
    fwd = fwd / fwd_norm
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    right_norm = np.linalg.norm(right)
    right = right / right_norm
    up = np.cross(right, fwd)
    aspect = float(WIDTH) / float(HEIGHT)
    tan_half_fov = math.tan(math.radians(fov_y_deg)/2)
    
    compute_pixels = np.zeros((COMPUTE_HEIGHT, COMPUTE_WIDTH, 3), dtype=np.uint8)
    
    print(f"Starting render at {COMPUTE_WIDTH}x{COMPUTE_HEIGHT}...")
    
    raytrace_kernel(compute_pixels, COMPUTE_WIDTH, COMPUTE_HEIGHT,
                    cam_pos, right, up, fwd,
                    tan_half_fov, aspect, disk_r1, disk_r2, SagA_rs,
                    BACKGROUND_IMAGE_DATA, BACKGROUND_WIDTH, BACKGROUND_HEIGHT,
                    DISK_IMAGE_DATA, DISK_WIDTH, DISK_HEIGHT, disk_opacity)
    
    print("Render complete. Upscaling...")
    
    img_low = Image.fromarray(compute_pixels, 'RGB')
    img_high = img_low.resize((WIDTH, HEIGHT), resample=Image.NEAREST)
    
    base_filename = "black_hole"
    extension = ".png"
    counter = 1
    output_filename = f"{base_filename}{extension}"
    while os.path.exists(output_filename):
        output_filename = f"{base_filename}_{counter}{extension}"
        counter += 1
    
    img_high.save(output_filename)
    print(f"Image saved to {output_filename}")
    img_high.show()