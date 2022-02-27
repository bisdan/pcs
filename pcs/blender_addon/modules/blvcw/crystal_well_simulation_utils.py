"""
Collection of utility functions used by other classes
"""

import math
import bpy_extras

import numpy as np
from numpy.random import default_rng


def get_normal_distributed_values(mu=0.0, sigma=1.0, number_values=1):
    rng = default_rng()
    return rng.normal(loc=mu,
                      scale=sigma,
                      size=number_values)


def get_uniformly_distributed_values(low=0.0, high=1.0, number_values=1):
    rng = default_rng()
    return rng.uniform(low=low,
                       high=high,
                       size=number_values)


def get_random_translation(gamma, camera_distance, field_of_view):
    f = np.random.random(3)
    x = np.power((3 / gamma) * f[2] + camera_distance ** 3, 1. / 3) - camera_distance
    l = 2 * (np.tan(field_of_view / 2) * (camera_distance + x))
    return l * (-.5 + f[0]), l * (-.5 + f[1]), -x


def get_random_x_y_translation(gamma, camera_distance, cw_depth, field_of_view):
    f = np.random.random(3)
    x = np.power((3 / gamma) * f[2] + camera_distance ** 3, 1. / 3) - camera_distance
    l = 2 * (np.tan(field_of_view / 2) * (camera_distance + x))
    return l * (-.5 + f[0]), l * (-.5 + f[1]), cw_depth


def get_random_normal_values_vector(averages=(0.0, 0.0, 0.0), std_dev=1.0, number_values=1):
    vectors = {"x": [],
               "y": [],
               "z": []}
    axis = "x"
    for axis_average in averages:
        vectors[axis].extend(get_normal_distributed_values(mu=axis_average,
                                                           sigma=std_dev,
                                                           number_values=number_values))
        axis = get_next_axis(axis=axis)
    return vectors


def get_random_uniform_values_vector(low=(0.0, 0.0, 0.0), high=(0.0, 0.0, 0.0), number_values=1):
    vectors = {"x": [],
               "y": [],
               "z": []}
    axis = "x"
    for i, axis_low in enumerate(low):
        axis_high = high[i]
        vectors[axis].extend(get_uniformly_distributed_values(low=axis_low,
                                                              high=axis_high,
                                                              number_values=number_values))
        axis = get_next_axis(axis=axis)
    return vectors


def get_random_euler(target_shape):
    return 0, 0, math.radians(target_shape["angle"])


def correct_angle(theta):
    return ((theta + 90) % -180) + 90


def get_next_axis(axis="x"):
    if axis == "x":
        return "y"
    if axis == "y":
        return "z"
    if axis == "z":
        return None
