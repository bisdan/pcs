import bpy
import bpy_extras
import math
import random
import cv2
import bmesh
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from mathutils import Euler, Vector

from abc import ABC, abstractmethod

from blvcw.crystal_well_simulation_utils import get_random_euler, get_normal_distributed_values, \
    get_random_normal_values_vector
from blvcw.crystal_well_components import CrystalWellLoader


class CrystalWellDistributor(ABC):
    def __init__(self):
        self.polygons = []

    @abstractmethod
    def get_crystals(self):
        pass

    def _reset(self):
        self.polygons = []


class CrystalWellRandomDistributor(CrystalWellDistributor):
    """
    Fast random distributor for crystals.
    Creates crystals with normal distributed values for scaling and rotation calculated with the given inputs
    for average and standard deviation.
    After that, places the new crystal randomly above the plane.
    """

    def __init__(self, number_crystals=20, scaling_crystals_average=(1.0, 1.0, 1.0), scaling_crystals_std_dev=0.0,
                 rotation_crystals_average=(0.0, 0.0, 0.0), rotation_crystals_std_dev=0.0, res_x=500, res_y=500,
                 random_translation_function=None, crystal_well_loader=None):
        super().__init__()

        self.number_crystals = number_crystals

        self.scaling_crystals_average = scaling_crystals_average
        self.scaling_crystals_std_dev = scaling_crystals_std_dev
        self.rotation_crystals_average = rotation_crystals_average
        self.rotation_crystals_std_dev = rotation_crystals_std_dev

        self.res_x = res_x
        self.res_y = res_y

        self.random_translation_function = random_translation_function
        self.crystal_well_loader: CrystalWellLoader = crystal_well_loader
        self.polygons = []

    def get_crystals(self):
        self._reset()

        camera = bpy.context.scene.camera

        # Scaling:
        scaling_vectors = get_random_normal_values_vector(averages=self.scaling_crystals_average,
                                                          std_dev=self.scaling_crystals_std_dev,
                                                          number_values=self.number_crystals)
        # Rotation:
        rotation_vectors = get_random_normal_values_vector(averages=self.rotation_crystals_average,
                                                           std_dev=self.rotation_crystals_std_dev,
                                                           number_values=self.number_crystals)
        for i in range(self.number_crystals):
            scaling_vector = (scaling_vectors["x"][i], scaling_vectors["y"][i], scaling_vectors["z"][i])
            rotation_vector = (rotation_vectors["x"][i], rotation_vectors["y"][i], rotation_vectors["z"][i])

            crystal = self.crystal_well_loader.load_crystal()
            crystal.location = self.random_translation_function()
            crystal.scale = scaling_vector
            crystal.rotation_euler = rotation_vector

            coords2d = project_solve(crystal, self.res_x, self.res_y, camera)
            self.polygons.append(coords2d.tolist())
            yield crystal

    def get_annotations(self):
        return self.polygons  # segmentation


class VCWDefaultDistributor(CrystalWellDistributor):

    def __init__(self, total_crystal_area_min=0.05, total_crystal_area_max=0.5, res_x=384, res_y=384,
                 max_n_crystals=500, crystal_area_min=3 ** 2, crystal_area_max=128 ** 2, crystal_edge_min=3,
                 crystal_edge_max=384, crystal_aspect_ratio_max=8 / 0.7, elongation_perturbation=6.0,
                 subdivide_edges=10, smooth_shading=True, random_translation_function=None, crystal_well_loader=None,
                 cw_depth=None):

        super().__init__()

        self.total_crystal_area_min = total_crystal_area_min
        self.total_crystal_area_max = total_crystal_area_max
        self.max_n_crystals = max_n_crystals

        self.crystal_area_min = crystal_area_min
        self.crystal_area_max = crystal_area_max
        self.crystal_edge_min = crystal_edge_min
        self.crystal_edge_max = crystal_edge_max
        self.crystal_aspect_ratio_max = crystal_aspect_ratio_max

        self.res_x = res_x
        self.res_y = res_y
        self.image_area = self.res_x * self.res_y
        self.elongation_perturbation = elongation_perturbation

        self.crystal_well_loader = crystal_well_loader
        self.random_translation_function = random_translation_function

        self.subdivide_edges = subdivide_edges
        self.smooth_shading = smooth_shading

        self.cw_depth = cw_depth  # camera data is not supposed to change during distribution of crystals and rendering

        self.polygons = []  # to collect annotations

    def _draw_area(self, area_min, area_max):
        return area_min * ((area_max / area_min) ** random.random())

    def _get_edge_lengths(self, area, ratio):
        return math.sqrt(area * ratio), math.sqrt(area / ratio)

    def _get_target_shape(self, remaining_crystal_area):
        area_max = min(self.crystal_area_max, remaining_crystal_area)

        if area_max < self.crystal_area_min:
            return None

        target_area = self._draw_area(self.crystal_area_min, area_max)
        aspect_ratio_max = min(self.crystal_aspect_ratio_max, (self.crystal_edge_max ** 2) / target_area)
        target_ratio = random.uniform(1, aspect_ratio_max)
        target_angle = random.uniform(-90, 90)
        return target_area, target_ratio, target_angle

    def _get_remaining_crystal_area(self, target_crystal_area, segmentations):
        if len(segmentations) == 0:
            return target_crystal_area
        else:
            return target_crystal_area - (np.sum(segmentations, axis=0) > 0).sum()

    def _elongation_perturbation(self, crystal):

        dx = random.uniform(0, self.elongation_perturbation)
        for vert in crystal.data.vertices:
            if vert.co[0] > 0:
                vert.co[0] += dx
            else:
                vert.co[0] -= dx

    def _to_camera_coords(self, crystal):
        # inspired by bpy_extras.object_utils.world_to_camera_view
        bpy.context.view_layer.update()
        co_local = [self.inv_mat @ (crystal.matrix_world @ v.co) for v in crystal.data.vertices]
        frames = [[-(v / (v.z / -co.z)) for v in self.frame] for co in co_local]

        min_xs, max_xs = [f[2].x for f in frames], [f[1].x for f in frames]
        min_ys, max_ys = [f[1].y for f in frames], [f[0].y for f in frames]

        xs = [(co.x - min_x) / (max_x - min_x) for co, min_x, max_x in zip(co_local, min_xs, max_xs)]
        ys = [(co.y - min_y) / (max_y - min_y) for co, min_y, max_y in zip(co_local, min_ys, max_ys)]

        res1 = np.array([[x * self.res_x, y * self.res_y, -co.z] for x, y, co in zip(xs, ys, co_local)], dtype=np.float32)

        """
        Test whether the all matrix formulation holds
        
        pmat = Matrix()
        pmat[0][2] = self.dpos[0] / self.dpos[2]
        pmat[1][2] = self.dpos[1] / self.dpos[2]
        pmat[2][2] = 1.0 / self.dpos[2]

        co_local2 = np.array([pmat @ self.inv_mat @ (crystal.matrix_world @ v.co) for v in crystal.data.vertices])

        # converting to normal coordinates
        res2 = np.array([[co[0] / co[2], co[1] / co[2], co[2]] for co in co_local2])
        res2[:, :2] *= 384

        assert np.allclose(res1[:,:2], res2[:,:2], rtol=1e-4)
        """
        return res1


    def _get_min_area_rect(self, crystal):
        cam_coords = self._to_camera_coords(crystal)
        (_, (width, height), theta) = cv2.minAreaRect(cam_coords[:, :2].copy())
        if width < height:
            width, height = height, width
            theta += 90
        theta = self._correct_angle(theta)
        return width, height, theta

    def _get_initial_guess(self, target_shape, crystal):
        target_area, target_ratio, target_angle = target_shape

        # scale estimation
        width, height, _ = self._get_min_area_rect(crystal)
        area = width * height
        ratio = width / height

        lam = math.sqrt((target_area * target_ratio) / (area * ratio))
        mu = (ratio / target_ratio) * lam

        angle = math.radians(target_angle)

        return np.array([lam, mu, angle], dtype=np.float32)

    def _correct_angle(self, theta):
        return ((theta + 90) % -180) + 90

    def _apply_params_area(self, crystal, lam, mu):
        h = math.sqrt(lam / (mu + 1e-4))
        w = mu * h
        crystal.scale = (w, h, h)

    def _apply_params_ratio(self, crystal, mu, lam):
        h = math.sqrt(lam / (mu + 1e-4))
        w = mu * h
        crystal.scale = (w, h, h)

    def _apply_params_angle(self, crystal, angle):
        crystal.rotation_euler = Euler((crystal.rotation_euler[0], crystal.rotation_euler[1], angle))

    def _l1_area(self, x, target_shape, crystal, mu):
        target_area, _, _ = target_shape
        self._apply_params_area(crystal, x, mu)

        width, height, _ = self._get_min_area_rect(crystal)
        area = width * height

        l1_edge = abs(math.sqrt(area) - math.sqrt(target_area))

        return l1_edge

    def _l1_ratio(self, x, target_shape, crystal, lam):
        _, target_ratio, _ = target_shape
        self._apply_params_ratio(crystal, x, lam)

        width, height, _ = self._get_min_area_rect(crystal)
        ratio = width / (height + 1e-4)

        l1_ratio = abs(ratio - target_ratio)

        return l1_ratio

    def _l1_angle(self, x, target_shape, crystal):
        _, _, target_angle = target_shape
        self._apply_params_angle(crystal, x)

        _, _, theta = self._get_min_area_rect(crystal)

        l1_angle = abs(target_angle - theta)
        if l1_angle > 90:
            l1_angle = 180 - l1_angle

        l1_angle /= 90.0

        return l1_angle


    def _is_close(self, target_shape, crystal):
        target_area, target_ratio, target_angle = target_shape
        width, height, theta = self._get_min_area_rect(crystal)
        area = width * height
        ratio = width / (height + 1e-4)
        da = abs(math.sqrt(target_area) - math.sqrt(area))
        dr = abs(target_ratio - ratio)
        dt = abs(target_angle - theta)
        if dt > 90:
            dt = abs(180 - dt)

        area_passed = da < 2
        ratio_passed = dr < 0.2
        angle_passed = dt < 2

        if area_passed and ratio_passed and angle_passed:
            return True, area_passed, ratio_passed, angle_passed
        else:
            return False, area_passed, ratio_passed, angle_passed


    def _in_bounds(self, crystal, cam_coords):
        xy_in_bounds = np.all(
            np.logical_and.reduce(
                (
                    cam_coords[:, 0] < self.res_x,
                    cam_coords[:, 0] >= 0,
                    cam_coords[:, 1] < self.res_y,
                    cam_coords[:, 1] >= 0,
                )
            )
        )
        if not xy_in_bounds:
            return False
        z_min = math.inf
        z_max = -math.inf
        for v in crystal.data.vertices:
            z_min = min(z_min, (crystal.matrix_world @ v.co)[2])
            z_max = max(z_max, (crystal.matrix_world @ v.co)[2])
        if z_min < self.cw_depth or z_max > 0:
            return False
        else:
            return True

    def _has_overlap(self, cam_coords, segmentations, iou_threshold=0.7):
        hull = cv2.convexHull(cam_coords[:, :2].copy())
        contour = np.int32(np.around(hull))
        this_segm = np.zeros((self.res_y, self.res_x), dtype=np.uint8)
        cv2.drawContours(this_segm, [contour], 0, 1, cv2.FILLED)

        if len(segmentations) == 0:
            return False, this_segm

        for segm in segmentations:
            n_overlap = np.logical_and(this_segm, segm).sum()
            n_union = np.logical_or(this_segm, segm).sum()
            iou = n_overlap / n_union
            if iou > iou_threshold:
                return True, None

        summed_segm = np.sum(segmentations, axis=0) + this_segm
        if summed_segm.max() > 3:
            return True, None
        else:
            return False, this_segm

    def _minimize_step(self, current_shape, crystal, target_shape):

        res_area = minimize_scalar(
            self._l1_area,
            args=(target_shape, crystal, current_shape[1]),
            bounds=(0, 1),
            method="Bounded",
            options={'xatol': 1e-6, 'maxiter': 500}
        )
        if not res_area.success:
            return None

        res_ratio = minimize_scalar(
            self._l1_ratio,
            args=(target_shape, crystal, res_area.x),
            bounds=(0.1, 20),
            method="Bounded",
            options={'xatol': 1e-6, 'maxiter': 500}
        )
        if not res_area.success:
            return None

        res_angle = minimize_scalar(
            self._l1_angle,
            args=(target_shape, crystal),
            bounds=(-math.pi, math.pi),
            method="Bounded",
            options={'xatol': 1e-6, 'maxiter': 500}
        )

        if not res_angle.success:
            return None
        else:
            return np.array((res_area.x, res_ratio.x, res_angle.x))

    def _minimize(self, current_shape, crystal, target_shape):
        max_tries = 4
        n_tries = 0
        while n_tries < max_tries:
            current_shape = self._minimize_step(current_shape, crystal, target_shape)
            if current_shape is None:
                return None
            flag, apass, rpass, tpass = self._is_close(target_shape, crystal)
            if flag:
                return current_shape, apass, rpass, tpass
            else:
                n_tries += 1

        return None, apass, rpass, tpass


    def _solve_for_target_shape(self, target_shape, crystal, segmentations, gather_stats=True):
        # requires initial crystals to be centered geometrically

        n_tries = 0
        while n_tries < 50:
            # print(n_tries)
            translation = self.random_translation_function()
            crystal.scale = (1, 1, 1)
            crystal.location = Vector((translation[0], translation[1], translation[2]))
            crystal.rotation_euler = (random.random() * math.pi, random.random() * math.pi, 0)

            initial_guess = self._get_initial_guess(target_shape, crystal)

            res, apass, rpass, tpass = self._minimize(initial_guess, crystal, target_shape)
            if res is None:
                n_tries += 1
                print("reject proposal: failed to minimize", n_tries, "area", apass, "ratio", rpass, "angle", tpass)
                continue

            cam_coords = self._to_camera_coords(crystal)
            if not self._in_bounds(crystal, cam_coords):
                n_tries += 1
                print("reject proposal: out of bounds", n_tries)
                continue

            has_overlap, segm = self._has_overlap(cam_coords, segmentations)
            if has_overlap:
                n_tries += 1
                print("reject proposal: has overlap", n_tries)
                continue
            else:
                segmentations.append(segm)

            return crystal, cam_coords

        return crystal, None

    def _setup_camera_inv_mat(self):
        camera = bpy.context.scene.camera
        assert camera.type != 'ORTHO'
        self.inv_mat = camera.matrix_world.normalized().inverted()
        self.frame = [v for v in camera.data.view_frame(scene=bpy.context.window.scene)[:3]]

    def _postprocessing(self, crystal):
        if self.subdivide_edges > 0:
            me = crystal.data
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, use_grid_fill=True, cuts=self.subdivide_edges)
            bm.to_mesh(me)
            bm.free()
            me.update()
        crystal.data.polygons.foreach_set('use_smooth', [self.smooth_shading] * len(crystal.data.polygons))
        crystal.data.update()
        return crystal

    def get_crystals(self):
        self._reset()
        self._setup_camera_inv_mat()

        segmentations = []

        target_total_crystal_area = random.uniform(self.total_crystal_area_min,
                                                   self.total_crystal_area_max) * self.image_area

        while len(self.polygons) < self.max_n_crystals:
            remaining_crystal_area = self._get_remaining_crystal_area(target_total_crystal_area, segmentations)
            target_shape = self._get_target_shape(remaining_crystal_area)
            if target_shape is None:
                # target total crystal area is reached
                break

            crystal = self.crystal_well_loader.load_crystal()
            self._elongation_perturbation(crystal)

            solved_crystal, cam_coords = self._solve_for_target_shape(target_shape, crystal, segmentations)
            if solved_crystal is None or cam_coords is None:
                # unsuccessful in placing the crystal in the current scene under the given constraints
                # try again with a different target shape
                # should be avoided since this distorts the target distributions in the final dataset.
                print("WARNING: Unable to place crystal within target shape: ", target_shape)
                bpy.data.objects.remove(crystal)
            else:
                print("Successfully placed crystal", target_shape)
                self.polygons.append(cam_coords.tolist())
                solved_crystal = self._postprocessing(solved_crystal)
                yield solved_crystal

    def get_annotations(self):
        return self.polygons
