import numpy as np

def lookat_matrix(origin, look_at, up):
    origin = np.array(origin)
    look_at = np.array(look_at)
    up = np.array(up)

    forward = look_at - origin
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = -right
    camera_matrix[:3, 1] = up
    camera_matrix[:3, 2] = forward

    camera_matrix[:3, 3] = origin

    return camera_matrix

    
def invert_camera_matrix(camera_matrix):
    camera_matrix = np.array(camera_matrix)
    
    R = camera_matrix[:3, :3]
    t = camera_matrix[:3, 3]
    
    R_inv = R.T
    
    t_inv = -R_inv @ t
    
    camera_matrix_inv = np.eye(4)
    camera_matrix_inv[:3, :3] = R_inv
    camera_matrix_inv[:3, 3] = t_inv
    
    return camera_matrix_inv