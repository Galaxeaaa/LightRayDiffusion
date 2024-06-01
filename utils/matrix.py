import numpy as np


def lookat_matrix(origin, look_at, up):
    origin = np.array(origin, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

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


def get_camera_intrinsics(fov):
    f = 1 / np.tan(np.deg2rad(fov) / 2)
    return np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
