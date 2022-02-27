import math
import random

import cv2
import numpy as np
import pyfastnoisesimd as fns
from scipy.interpolate import CubicSpline

from pcs.transforms import *

from pcs.utils import (
    get_rotation_matrix, 
    grayscale_to_float,
    grayscale_to_uint
)

def _randomize_structured_noise_frequency(shape):
    # from a few pixels
    scale_min = 4
    # to a diagonal spanning the input image dimensions
    scale_max = math.sqrt(shape[0]**2 + shape[1]**2)

    assert scale_min < scale_max and scale_min > 0
    exponent = random.uniform(
        -math.log(scale_max) / math.log(scale_min),
        -1
    )
    return scale_min ** exponent


class PCSAugmentation:
    def get_transform(self, image, segmentations, rotated_boxes):
        raise NotImplementedError

class PCSNoOpAugmentation(PCSAugmentation):
    def get_transform(self, image, segmentations, rotated_boxes):
        return PCSNoOpTransform()


class PCSAugmentationList(PCSAugmentation):
    def __init__(self, augmentations):
        auglist = []
        for augmentation in augmentations:
            if isinstance(augmentation, PCSAugmentation):
                auglist.append(augmentation)
        self.augmentations = auglist

    def __call__(self, image, boxes, sem_seg):
        transforms = [
            augmentation.get_transform(image, boxes, sem_seg)
            for augmentation in self.augmentations
        ]
        return PCSTransformList(transforms)

class PCSPerlinAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.8, max_num_layers=10):
        self.p_image = p_image
        self.max_num_layers = max_num_layers

    @staticmethod
    def _generate_perlin_layer(shape, frequency, n_threads=4):
        # https://github.com/robbmcleod/pyfastnoisesimd
        seed = random.randint(0, 2 ** 31)
        perlin = fns.Noise(seed=seed, numWorkers=n_threads)
        perlin.frequency = frequency
        perlin.noiseType = fns.NoiseType.Perlin
        # The following settings are currently the default settings for the pyfastnoisesimd package.
        # Set them anyways in case they change at some point.
        perlin.perturb.perturbType = fns.PerturbType.NoPerturb
        perlin.fractal.octaves = 4
        perlin.fractal.lacunarity = 2

        result = perlin.genAsGrid(shape)
        # map values from [-1, 1] to [0, 1]
        result = (result + 1) / 2

        # normalize to [0, 1]
        result -= result.min()
        # This case should (almost) never happen
        if result.max() == 0:
            return None
        result /= result.max()

        # equalize for consistent behavior
        result = grayscale_to_float(
            cv2.equalizeHist(
                grayscale_to_uint(result)
            )
        )

        return result

    @staticmethod
    def _stack_perlin_layers(
        shape,
        max_num_layers=10,
    ):
        num_layers = random.randint(1, max_num_layers)
        frequencies = [
            _randomize_structured_noise_frequency(shape)
            for i in range(num_layers)
        ]        
        layers = [
            PCSPerlinAugmentation._generate_perlin_layer(shape, freq)
            for freq in frequencies
        ]
        layer = np.mean(layers, axis=0)

        # normalize to [0, 1]
        layer -= layer.min()
        # This case should (almost) never happen
        if layer.max() == 0:
            return None
        layer /= layer.max()

        # equalize for consistent behavior
        layer = grayscale_to_float(
            cv2.equalizeHist(
                grayscale_to_uint(layer)
            )
        )
        return layer

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()
        
        layer_shape = image.shape

        base_layer = PCSPerlinAugmentation._stack_perlin_layers(
            layer_shape,
            max_num_layers=self.max_num_layers
        )
        alpha_layer = PCSPerlinAugmentation._stack_perlin_layers(
            layer_shape,
            max_num_layers=self.max_num_layers
        )

        if base_layer is None or alpha_layer is None:
            return PCSNoOpTransform()

        return PCSBlendTransform(
            base_layer=base_layer,
            alpha_layer=alpha_layer,
            suppress=random.choice(("bright", "dark"))
        )

class PCSBrightnessAugmentation(PCSAugmentation):
    def __init__(self, p_image=1.0):
        self.p_image = p_image

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        enhancement_factor = 2 ** random.uniform(-1, 1)

        return PCSBrightnessTransform(enhancement_factor=enhancement_factor)


class PCSGaussNoiseAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.5, std_lower=0.01, std_upper=0.06):
        self.std_lower = std_lower
        self.std_upper = std_upper
        self.p_image = p_image

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        noise_scale = random.uniform(self.std_lower, self.std_upper)
        gauss_noise = np.random.normal(
            loc=0, scale=noise_scale, size=image.shape
        )
        return PCSAdditiveTransform(gauss_noise)

class PCSImpulsNoiseAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.2, p_salt=0.05, p_pepper=0.05):
        self.p_salt = p_salt
        self.p_pepper = p_pepper
        self.p_image = p_image

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        p_salt = random.uniform(0, self.p_salt)
        p_pepper = random.uniform(0, self.p_pepper)

        p = np.random.random(image.shape)
        mask_salt = p > 1 - p_salt
        mask_pepper = p < p_pepper

        mask = np.full(image.shape, 0, dtype=np.float32)
        mask[mask_salt] = -1
        mask[mask_pepper] = 1

        return PCSAdditiveTransform(
            mask=mask
        )


class PCSWaveAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.2, max_num_layers=10, max_phase_offset=384):
        self.p_image = p_image
        self.max_num_layers = max_num_layers
        self.max_phase_offset = max_phase_offset

    @staticmethod
    def _random_wave_vector(shape):
        theta = random.uniform(0, 2 * math.pi)
        freq = 2 * math.pi * _randomize_structured_noise_frequency(shape)
        return (
            freq * math.sin(theta),
            freq * math.cos(theta)
        )


    def _get_wave_layer(self, xx, yy, prop):

        phase = random.uniform(0, 2*math.pi)

        layer = np.cos(yy * prop[0] + xx * prop[1] + phase)

        layer -= layer.min()
        if layer.max() == 0:
            return None      
        layer /= layer.max()
        
        # equalize intensity for consistent behavior
        layer = grayscale_to_float(
            cv2.equalizeHist(
                grayscale_to_uint(layer)
            )
        )

        return layer

    def _stack_wave_layers(self, image):
        height, width = image.shape
        xx, yy = np.meshgrid(
            np.arange(height),
            np.arange(width)
        )

        propagations = np.array(
            [
                PCSWaveAugmentation._random_wave_vector(image.shape)
                for i in range(random.randint(1, self.max_num_layers))
            ]
        )

        layers = [
            self._get_wave_layer(xx, yy, prop) for prop in propagations
        ]
        layer = np.mean(layers, axis=0)


        layer -= layer.min()
        if layer.max() == 0:
            return None      
        layer /= layer.max()
        
        # equalize intensity for consistent behavior
        layer = grayscale_to_float(
            cv2.equalizeHist(
                grayscale_to_uint(layer)
            )
        )
        return layer

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        base_layer = self._stack_wave_layers(image)
        alpha_layer = self._stack_wave_layers(image)

        if base_layer is None or alpha_layer is None:
            return PCSNoOpTransform()

        return PCSBlendTransform(
            base_layer=base_layer,
            alpha_layer=alpha_layer,
            suppress=random.choice(("bright", "dark"))
        )


class PCSCompressionAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.5, quality_lower=60, quality_upper=95):
        super().__init__()
        self.p_image = p_image
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        quality = random.randint(self.quality_lower, self.quality_upper)
        return PCSCompressionTransform(quality=quality)


class PCSBlurAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.5, possible_kernel_sizes=[3, 5, 7, 9]):
        self.p_image = p_image
        self.possible_kernel_sizes = possible_kernel_sizes

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()
        kernel_size = random.choice(self.possible_kernel_sizes)
        return PCSBlurringTransform(kernel_size=kernel_size)


class PCSDistortionAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.1):
        self.p_image = p_image

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()
        else:
            shift_lambda = lambda x: int(random.choice((-1, 0, 1)))
            return PCSDistortionTransform(shift_lambda=shift_lambda)


class PCSFilterUndetectableAugmentation(PCSAugmentation):
    def __init__(self, min_contrast_variation=0.0):
        super().__init__()
        self.min_contrast_variation = min_contrast_variation

    def get_transform(self, image, segmentations, rotated_boxes):
        assert len(segmentations) == len(rotated_boxes)
        return PCSFilterUndetectableTransform(
            num_objects=len(segmentations),
            min_contrast_variation=self.min_contrast_variation,
        )

class PCSFilterInvalidAugmentation(PCSAugmentation):
    def __init__(self, min_edge_length=0):
        super().__init__()
        self.min_edge_length = min_edge_length

    def get_transform(self, image, segmentations, rotated_boxes):
        assert len(segmentations) == len(rotated_boxes)
        return PCSFilterInvalidTransform(
            num_objects=len(segmentations), min_edge_length=self.min_edge_length
        )


class PCSFlipAugmentation(PCSAugmentation):

    def get_transform(self, image, segmentations, rotated_boxes):
        p = np.random.random()
        flip = random.choice((None, "h", "v"))
        if flip:
            return PCSFlipTransform(flip=flip)
        else:
            return PCSNoOpTransform()


class PCSRotationAugmentation(PCSAugmentation):

    def get_transform(self, image, segmentations, rotated_boxes):
        theta_degree = np.random.choice((0, 90, 180, 270))
        if theta_degree == 0:
            return PCSNoOpTransform()
        else:
            return PCSRotationTransform(theta_degree=theta_degree)


class PCSSplineAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.2, max_num_splines=10, max_num_spline_points=4):
        self.p_image = p_image
        self.max_num_splines = max_num_splines
        self.max_num_spline_points = max_num_spline_points

    @staticmethod
    def _get_random_spline(shape, max_num_spline_points=4):
        height, width = shape
        n_points = random.randint(2, max_num_spline_points)
        xs = np.linspace(-0.5, 0.5, n_points)
        ys = np.random.random(n_points) - 0.5
        cs = CubicSpline(xs, ys)
        xs = np.linspace(xs.min(), xs.max(), 100)
        ys = cs(xs)
        xs = np.array(xs) * width
        ys = np.array(ys) * height
        coords = np.vstack((xs, ys)).T
        rotmat = get_rotation_matrix(random.uniform(0, 360))
        coords = np.dot(rotmat, coords.T).T
        
        center = np.array(
            (
                height / 2,
                width / 2
            )
        )
        coords += center
        
        return np.around(coords).astype(np.int32)


    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        n_splines = random.randint(1, self.max_num_splines)

        height, width = image.shape

        mask_shape = (
            2 * height,
            2 * width,
        )

        mask = np.full(mask_shape, 128, dtype=np.uint8)

        splines = [
            PCSSplineAugmentation._get_random_spline(
                mask_shape,
                max_num_spline_points=self.max_num_spline_points
            )
            for i in range(n_splines)
        ]

        for spline in splines:
            cv2.polylines(
                mask,
                [spline],
                False,
                random.randint(0, 255),
                random.randint(1, 3),
                cv2.LINE_AA,
            )

        
        mask = cv2.resize(
            mask, image.shape, interpolation=cv2.INTER_AREA
        )

        # shift to [-1, 1]
        mask = 2 * (grayscale_to_float(mask) - 0.5)
        return PCSAdditiveTransform(mask=mask)


class PCSCircularOverlayAugmentation(PCSAugmentation):
    def __init__(self, p_image=0.1):
        self.p_image = p_image

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        mask = np.full_like(image, 255, dtype=np.uint8)

        if random.random() > 0.5:
            radius = image.shape[0] + int(
                round(random.uniform(image.shape[0] * 0.8 / 2, image.shape[0] * 1.2 / 2))
            )
            center = random.choice(
                (
                    [0, 0],
                    [0, mask.shape[1]],
                    [mask.shape[0], 0],
                    [mask.shape[0], mask.shape[1]],
                )
            )
            center[0] += int(round((random.random() - 0.5) * image.shape[0] * 0.2))
            center[1] += int(round((random.random() - 0.5) * image.shape[1] * 0.2))
            cv2.circle(mask, tuple(center), radius, 0, thickness=200)
        else:
            ox = random.randint(-31, 31)
            oy = random.randint(-31, 31)
            if oy < 0:
                mask[oy:,:] = 0
            else:
                mask[:oy,:] = 0
            if ox < 0:
                mask[ox:,:] = 0
            else:
                mask[:ox,:] = 0

        kernel_size = random.randint(0, 15)
        if kernel_size > 0:
            kernel_size = kernel_size * 2 + 1
            mask = cv2.GaussianBlur(
                mask,
                (
                    kernel_size,
                    kernel_size,
                ),
                cv2.BORDER_REFLECT,
            )
        mask = grayscale_to_float(mask) - 1
        return PCSAdditiveTransform(mask=mask)


class PCSContrastAugmentation(PCSAugmentation):
    def __init__(self, p_image=1.0):
        self.p_image = p_image

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpAugmentation()
        else:
            enhancement_factor = 1.5 ** random.uniform(0, 1)
            return PCSContrastTransform(enhancement_factor=enhancement_factor)

class PCSHighlightAugmentation(PCSAugmentation):
    def __init__(
        self,
        p_image=1.0,
        p_object=1,
        highlight_lower=90,
        highlight_upper=255-90
    ):
        self.p_image = p_image
        self.p_object = p_object
        self.highlight_lower = highlight_lower
        self.highlight_upper = highlight_upper

    def get_transform(self, image, segmentations, rotated_boxes):
        if random.random() > self.p_image:
            return PCSNoOpTransform()

        assert len(segmentations) == len(rotated_boxes)

        segmentations = [
            np.int32(np.around(segmentation)) for segmentation in segmentations
        ]

        mask = np.full(image.shape, 128, dtype=np.uint8)
        for segmentation, box in zip(segmentations, rotated_boxes):
            if random.random() < 0.2:
                # highlight whole crystals
                highlight_color = random.randint(
                    self.highlight_lower,
                    self.highlight_upper
                )
                cv2.drawContours(
                    mask,
                    [segmentation],
                    0,
                    highlight_color,
                    -1
                )
                if random.random() < 0.5:
                    # highlight the same cystal with a displaced map
                    object_mask = np.full(
                        image.shape,
                        0,
                        dtype=np.uint8
                    )
                    cv2.drawContours(
                        object_mask,
                        [segmentation],
                        0,
                        1,
                        -1
                    )
                    object_mask = object_mask == 1
                    tmp = np.full(
                        image.shape,
                        128,
                        dtype=np.uint8
                    )
                    displacement = random.choice((1, -1)) * get_rotation_matrix(box[-1]).dot((0,1)) * box[-2]/2
                    displacement = displacement.astype(np.int32)
                    highlight_color = random.randint(
                        self.highlight_lower,
                        self.highlight_upper
                    )
                    cv2.drawContours(
                        tmp,
                        [segmentation + displacement],
                        0,
                        255 - highlight_color,
                        -1
                    )
                    mask[(tmp != 128) * object_mask] = tmp[(tmp != 128) * object_mask]
            elif random.random() < 0.2:
                # highlight only crystal edges
                highlight_color = random.randint(
                    self.highlight_lower,
                    self.highlight_upper
                )
                cv2.drawContours(
                    mask,
                    [segmentation],
                    0,
                    highlight_color,
                    random.randint(1, 2)
                )

        kernel_size = random.choice((3, 5,))

        mask = cv2.GaussianBlur(
            mask,
            (
                kernel_size,
                kernel_size,
            ),
            cv2.BORDER_REFLECT,
        )
        mask = grayscale_to_float(mask) - 0.5
        return PCSAdditiveTransform(mask=mask)



class PCSDefaultAugmentationList(PCSAugmentationList):
    def __init__(
        self,
        p_highlight=0.5,
        p_brightness=1.0,
        p_contrast=1.0,
        p_spline=0.2,
        p_distort=0.05,
        p_blur=0.7,
        p_gauss=0.4,
        p_impuls=0.2,
        p_wave=0.4,
        p_perlin=0.4,
        p_overlay=0.2
    ):
        augmentation_list = [
            PCSHighlightAugmentation(
                p_image=p_highlight
            ),
            PCSSplineAugmentation(
                p_image=p_spline
            ),
            PCSDistortionAugmentation(
                p_image=p_distort
            ),
            PCSGaussNoiseAugmentation(
                p_image=p_gauss
            ),
            PCSImpulsNoiseAugmentation(
                p_image=p_impuls
            ),
            PCSBrightnessAugmentation(
                p_image=p_brightness
            ),
            PCSContrastAugmentation(
                p_image=p_contrast
            ),
            PCSWaveAugmentation(
                p_image=p_wave
            ),
            PCSPerlinAugmentation(
                p_image=p_perlin
            ),
            PCSContrastAugmentation(
                p_image=p_contrast
            ),
            PCSCircularOverlayAugmentation(
                p_image=p_overlay
            ),
            PCSBlurAugmentation(
                p_image=p_blur
            ),
            PCSRotationAugmentation(),
            PCSFlipAugmentation()
        ]
        super().__init__(augmentation_list)


class PCSDefaultAugmentor:
    def __init__(
        self,
        min_edge_length=0,
        min_contrast_variation=0,
    ):
        self.index = 0
        self.min_edge_length = min_edge_length
        self.min_contrast_variation = min_contrast_variation

    def filter_invalid(self, image, segmentations, rotated_boxes):

        invalid_filter = PCSFilterInvalidAugmentation(
            min_edge_length=self.min_edge_length
        )
        invalid_filter = invalid_filter.get_transform(
            image, segmentations, rotated_boxes
        )
        rotated_boxes = [
            invalid_filter.apply_rotated_box(rotated_box)
            for rotated_box in rotated_boxes
        ]

        valid_segmentations = []
        valid_rotated_boxes = []

        for object_index, (segmentation, rotated_box) in enumerate(
            zip(segmentations, rotated_boxes)
        ):
            if invalid_filter.valid_flags[object_index]:
                valid_segmentations.append(segmentation)
                valid_rotated_boxes.append(rotated_box)

        return valid_segmentations, valid_rotated_boxes

    def filter_undetectable(self, image, segmentations, rotated_boxes):

        undetectable_filter = PCSFilterUndetectableAugmentation(
            min_contrast_variation=self.min_contrast_variation
        )
        undetectable_filter = undetectable_filter.get_transform(
            image, segmentations, rotated_boxes
        )
        undetectable_filter.apply_image(image)
        segmentations = [
            undetectable_filter.apply_segmentation(segmentation)
            for segmentation in segmentations
        ]

        detectable_segmentations = []
        detectable_rotated_boxes = []

        for object_index, (segmentation, rotated_box) in enumerate(
            zip(segmentations, rotated_boxes)
        ):
            if undetectable_filter.detectable_flags[object_index]:
                detectable_segmentations.append(segmentation)
                detectable_rotated_boxes.append(rotated_box)

        return detectable_segmentations, detectable_rotated_boxes

    def __call__(self, image, segmentations, rotated_boxes):
        valid_segmentations, valid_rotated_boxes = self.filter_invalid(
            image, segmentations, rotated_boxes
        )

        pcs_augmentation_list = PCSDefaultAugmentationList()

        pcs_transform_list = pcs_augmentation_list(
            image, valid_segmentations, valid_rotated_boxes
        )
        image = pcs_transform_list.apply_image(image)
        segmentations = [
            pcs_transform_list.apply_segmentation(segmentation)
            for segmentation in valid_segmentations if segmentation is not None
        ]
        rotated_boxes = [
            pcs_transform_list.apply_rotated_box(rotated_box)
            for rotated_box in valid_rotated_boxes if rotated_box is not None
        ]

        detectable_segmentations, detectable_rotated_boxes = self.filter_undetectable(
            image, segmentations, rotated_boxes
        )

        self.index += 1

        augmentation_result = dict(
            aug_img = image,
            aug_segms = detectable_segmentations,
            aug_rbboxs = detectable_rotated_boxes,
            augmentations = pcs_augmentation_list.augmentations,
            transforms = pcs_transform_list.transforms
        )

        return augmentation_result