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

def ensure_image_rgb(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("image is not an instance of 'numpy.ndarray'")

    # ensure that image is in range [0, 255]
    if np.issubdtype(image.dtype, np.floating):
        if not (image.max() <= 1 and image.min() >= 0):
            raise ValueError(f"image float values outside of expected range [0, 1]: [{image.min()}, {image.max()}]")
        image = grayscale_to_uint(image)
    elif np.issubdtype(image.dtype, np.integer):
        if not (image.max() <= 255 and image.min() >= 0):
            raise ValueError("image integer values outside of expected range [0, 255]")
    
    # ensure that image has rgb channels for colored drawing operations
    if len(image.shape) == 2:
        image = grayscale_to_rgb(image)
    elif not image.shape[2] == 3:
        raise ValueError("three color channels are expected for drawing operations")

    return image
