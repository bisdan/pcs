import math

import numpy as np


def correct_angle(theta):
    return ((theta + 90) % -180) + 90

def convert_min_area_rect(min_area_rect):
    ((x_pos, y_pos), (width, height), theta) = min_area_rect
    if width < height:
        width, height = height, width
        theta += 90
    return (x_pos, y_pos, width, height, -correct_angle(theta))

def get_rotation_matrix(theta_deg):
    theta = math.radians(theta_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    return np.array(((cos, -sin), (sin, cos)))

def get_corners(rotated_box, f=1.0):

    w2 = rotated_box[2] / 2
    h2 = rotated_box[3] / 2

    theta_degree = rotated_box[4]

    rot_mat = get_rotation_matrix(f * theta_degree)

    coords = np.array(((-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)))
    corners = rot_mat.dot(coords.T).T + np.array((rotated_box[0], rotated_box[1]))

    return corners

def grayscale_to_uint(image):
    return np.uint8(np.around(255 * image))

def grayscale_to_rgb(image):
    return np.repeat(image[:, :, np.newaxis], 3, axis=2)

def grayscale_to_float(image):
    return image / 255
