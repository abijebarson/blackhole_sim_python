###############################################
########### Numba Helper Functions ############
######## (Nanba in tamil means friend) ########
###############################################

import numpy as np
import math
from numba import jit, prange

STEPS = 60000
D_LAMBDA = 1e-3
ESCAPE_R = 1e6

@jit(nopython=True, fastmath=True, cache=True)
def geodesic_rhs(r, theta, dr, dtheta, dphi, E, rs):
    f = 1.0 - rs / r
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    dt_dL = E / f
    d1_r = dr
    d1_theta = dtheta
    d1_phi = dphi
    d2_r = (
        - (rs / (2.0 * r*r)) * f * dt_dL * dt_dL
        + (rs / (2.0 * r*r * f)) * dr * dr
        + r * f * (dtheta*dtheta + sin_theta*sin_theta*dphi*dphi) # Corrected f here
    )
    d2_theta = (
        - 2.0 * dr * dtheta / r 
        + sin_theta * cos_theta * dphi * dphi
    )
    d2_phi = -2.0 * dr * dphi / r
    if sin_theta > 1e-6:
        d2_phi -= 2.0 * cos_theta / sin_theta * dtheta * dphi
    return d1_r, d1_theta, d1_phi, d2_r, d2_theta, d2_phi

@jit(nopython=True, fastmath=True, cache=True)
def euler_step(r, theta, phi, dr, dtheta, dphi, E, L, rs, dL):
    # TODO: Also try rk4

    # Actual geodesic equations
    d1_r, d1_theta, d1_phi, d2_r, d2_theta, d2_phi = geodesic_rhs(
        r, theta, dr, dtheta, dphi, E, rs
    )
    r_new = r + d1_r * dL
    theta_new = theta + d1_theta * dL
    phi_new = phi + d1_phi * dL
    dr_new = dr + d2_r * dL
    dtheta_new = dtheta + d2_theta * dL
    dphi_new = dphi + d2_phi * dL
    sin_theta_new = math.sin(theta_new)
    x = r_new * sin_theta_new * math.cos(phi_new)
    y = r_new * sin_theta_new * math.sin(phi_new)
    z = r_new * math.cos(theta_new)
    return x, y, z, r_new, theta_new, phi_new, dr_new, dtheta_new, dphi_new

@jit(nopython=True, fastmath=True, cache=True)
def init_ray(pos, dir, rs):
    x, y, z = pos[0], pos[1], pos[2]
    dx, dy, dz = dir[0], dir[1], dir[2]
    r = math.sqrt(x*x + y*y + z*z)
    z_over_r = z / r
    if z_over_r > 1.0: z_over_r = 1.0
    if z_over_r < -1.0: z_over_r = -1.0
    theta = math.acos(z_over_r)
    phi = math.atan2(y, x)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    dr = sin_theta*cos_phi*dx + sin_theta*sin_phi*dy + cos_theta*dz
    dtheta = (cos_theta*cos_phi*dx + cos_theta*sin_phi*dy - sin_theta*dz) / r
    dphi = (-sin_phi*dx + cos_phi*dy) / (r * sin_theta) if sin_theta > 1e-6 else 0.0
    L = r * r * sin_theta * dphi 
    f = 1.0 - rs / r
    dt_dL_sq = (dr*dr)/f + r*r*(dtheta*dtheta + sin_theta*sin_theta*dphi*dphi)
    if dt_dL_sq < 0.0: dt_dL_sq = 0.0 
    dt_dL = math.sqrt(dt_dL_sq/f) # Correction here
    E = f * dt_dL
    return x, y, z, r, theta, phi, dr, dtheta, dphi, E, L

@jit(nopython=True, fastmath=True, cache=True)
def crosses_equatorial_plane(old_y, new_y, new_x, new_z, r1, r2):
    crossed = (old_y * new_y < 0.0)
    if not crossed:
        return False
    r = math.sqrt(new_x*new_x + new_z*new_z)
    return (r >= r1 and r <= r2)

@jit(nopython=True, fastmath=True, cache=True)
def sample_background_planar(lensed_dir_x, lensed_dir_y, lensed_dir_z,
                           cam_right, cam_up, cam_forward,
                           bg_data, bg_width, bg_height,
                           tan_half_fov, aspect):
    
    lensed_dir = np.array([lensed_dir_x, lensed_dir_y, lensed_dir_z])
    
    cam_space_x = np.dot(lensed_dir, cam_right)
    cam_space_y = np.dot(lensed_dir, cam_up)
    cam_space_z = np.dot(lensed_dir, cam_forward)
    
    if cam_space_z < 1e-6: 
        return (0, 0, 0) 

    u_proj = cam_space_x / cam_space_z
    v_proj = cam_space_y / cam_space_z
    
    scale = 0.5 
    u = (u_proj * scale) + 0.5
    v = (v_proj * scale) + 0.5
    
    px = int(u * bg_width) % bg_width
    py = int(v * bg_height) % bg_height
    
    return (bg_data[py, px, 0], bg_data[py, px, 1], bg_data[py, px, 2])

@jit(nopython=True, fastmath=True, cache=True)
def sample_disk_texture(hit_x, hit_z, disk_data, disk_width, disk_height, disk_r2):
    angle = math.atan2(hit_z, hit_x) 
    u = (angle + math.pi) / (2.0 * math.pi) 

    radial_dist = math.sqrt(hit_x*hit_x + hit_z*hit_z)
    v = radial_dist / disk_r2 
    
    px = int(u * disk_width) % disk_width
    py = int(v * disk_height) % disk_height

    return (disk_data[py, px, 0], disk_data[py, px, 1], disk_data[py, px, 2])


@jit(nopython=True, fastmath=True, cache=True)
def raywarp_pixel(px, py, compute_width, compute_height, 
                   cam_pos, cam_right, cam_up, cam_forward,
                   tan_half_fov, aspect, disk_r1, disk_r2, rs,
                   background_image_data, background_width, background_height,
                   disk_image_data, disk_width, disk_height, disk_opacity):
    
    u_ndc = (2.0 * (px + 0.5) / compute_width - 1.0) * aspect * tan_half_fov # normalized device coords (ndcs)
    v_ndc = (1.0 - 2.0 * (py + 0.5) / compute_height) * tan_half_fov
    
    # Compute the ray direction in world space (NOTE: manually inlined for numba performance)
    # dir_vec = (u_ndc * cam_right) - (v_ndc * cam_up) + cam_forward
    # dir_vec = dir_vec / np.linalg.norm(dir_vec)

    dir_x = u_ndc * cam_right[0] - v_ndc * cam_up[0] + cam_forward[0]
    dir_y = u_ndc * cam_right[1] - v_ndc * cam_up[1] + cam_forward[1]
    dir_z = u_ndc * cam_right[2] - v_ndc * cam_up[2] + cam_forward[2]
    
    dir_norm = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    dir_vec_x = dir_x / dir_norm
    dir_vec_y = dir_y / dir_norm
    dir_vec_z = dir_z / dir_norm

    dir_vec = np.array([dir_vec_x, dir_vec_y, dir_vec_z])

    (x, y, z, 
     r, theta, phi, 
     dr, dtheta, dphi, 
     E, L) = init_ray(cam_pos, dir_vec, rs)
    
    prev_y = y
    
    hit_black_hole = False
    
    final_r, final_g, final_b = 0.0, 0.0, 0.0
    remaining_light = 1.0

    for i in range(STEPS):
        if r <= rs:
            hit_black_hole = True
            break
            
        (x, y, z, r, theta, phi, 
         dr, dtheta, dphi) = euler_step(r, theta, phi, dr, dtheta, dphi, E, L, rs, D_LAMBDA)

        if crosses_equatorial_plane(prev_y, y, x, z, disk_r1, disk_r2):
            disk_r_col, disk_g_col, disk_b_col = sample_disk_texture(
                x, z, disk_image_data, disk_width, disk_height, disk_r2
            )
            
            final_r += disk_r_col * disk_opacity * remaining_light
            final_g += disk_g_col * disk_opacity * remaining_light
            final_b += disk_b_col * disk_opacity * remaining_light
            
            remaining_light *= (1.0 - disk_opacity)
            
            if remaining_light < 1e-6:
                break
            
        prev_y = y
        if r > ESCAPE_R:
            # print('something escaped')
            break
            
    if hit_black_hole:
        return (final_r, final_g, final_b)
        
    elif remaining_light < 1e-6:
          return (final_r, final_g, final_b)
          
    else: 
        final_dir_norm = math.sqrt(x*x + y*y + z*z)
        lensed_dir_x = x / final_dir_norm
        lensed_dir_y = y / final_dir_norm
        lensed_dir_z = z / final_dir_norm
        
        r_col, g_col, b_col = sample_background_planar(
            lensed_dir_x, lensed_dir_y, lensed_dir_z, 
            cam_right, cam_up, cam_forward,
            background_image_data, background_width, background_height,
            tan_half_fov, aspect
        )
        
        final_r += r_col * remaining_light
        final_g += g_col * remaining_light
        final_b += b_col * remaining_light
        return (final_r, final_g, final_b)

# My definitions:
# Raytrace - when the ray is traced analytically in straight lines until it hits something
# Raymarch - when the ray is traced in small steps, sampling along the way in straight lines
# Raywarp - when the ray is traced in small steps, but a geodesic for two points is no longer a straight line, so the rays warp around spacetime

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def raywarp_kernel(pixels, compute_width, compute_height,
                    cam_pos, cam_right, cam_up, cam_forward,
                    tan_half_fov, aspect, disk_r1, disk_r2, rs,
                    background_image_data, background_width, background_height,
                    disk_image_data, disk_width, disk_height, disk_opacity):
    for py in prange(compute_height):
        for px in range(compute_width):
            r, g, b = raywarp_pixel(px, py, compute_width, compute_height,
                                     cam_pos, cam_right, cam_up, cam_forward,
                                     tan_half_fov, aspect, disk_r1, disk_r2, rs,
                                     background_image_data, background_width, background_height,
                                     disk_image_data, disk_width, disk_height, disk_opacity)
            pixels[py, px, 0] = r
            pixels[py, px, 1] = g
            pixels[py, px, 2] = b