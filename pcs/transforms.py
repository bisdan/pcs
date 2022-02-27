import io

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from pcs.utils import (correct_angle, get_rotation_matrix, grayscale_to_float,
                       grayscale_to_uint)


def _weighted_blending(base_layer, weights1, alpha_layer, weights2):
    image = (base_layer * weights1 + alpha_layer * weights2) / (weights1 + weights2)
    return image

def _additive_blending(image, mask):
    image = image + mask
    image[image > 1] = 1
    image[image < 0] = 0
    return image

class PCSTransform:
    def __init__(self):
        pass
    def apply_image(self, image):
        return image
    def apply_coords(self, coords):
        return coords
    def apply_rotated_box(self, rotated_box):
        return rotated_box
    def apply_segmentation(self, segmentation):
        return segmentation

class PCSNoOpTransform(PCSTransform):
    pass

class PCSTransformListIterator:
    def __init__(self, transform_list):
        self.transform_list = transform_list
        self._index = 0

    def __next__(self):
        if self._index < len(self.transform_list.transforms):
            transform = self.transform_list.transforms[self._index]
            self._index += 1
            return transform
        else:
            raise StopIteration


class PCSTransformList(PCSTransform):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
    
    def apply_image(self, image):            
        for transform in self.transforms:
            image = transform.apply_image(image)
            assert image.max() <= 1 and image.min() >= 0
        
        return image
    
    def apply_rotated_box(self, rotated_box):
        for transform in self.transforms:
            rotated_box = transform.apply_rotated_box(rotated_box)
            if rotated_box is None:
                return None
        return rotated_box
    
    def apply_coords(self, coords):
        for transform in self.transforms:
            coords = transform.apply_coords(coords)
            if coords is None:
                return None
        return coords

    def apply_segmentation(self, segmentation):
        for transform in self.transforms:
            segmentation = transform.apply_segmentation(segmentation)
            if segmentation is None:
                return None
        return segmentation

    def __iter__(self):
        return PCSTransformListIterator(self)

class PCSBlendTransform(PCSTransform):
    def __init__(self, base_layer, alpha_layer, suppress):
        self.base_layer = base_layer
        self.alpha_layer = alpha_layer
        self.suppress = suppress

    def apply_image(self, image):
        nom = 5 * image.std() * self.alpha_layer
        den = 2 ** (2 * self.base_layer - 1)

        return _weighted_blending(
            image,
            1,
            self.base_layer,
            nom / den
        )

class PCSAdditiveTransform(PCSTransform):
    def __init__(self, mask):
        self.mask = mask
    def apply_image(self, image):
        return _additive_blending(image, self.mask)

class PCSBrightnessTransform(PCSTransform):
    def __init__(self, enhancement_factor):
        self.enhancement_factor = enhancement_factor

    def apply_image(self, image):
        image = grayscale_to_uint(image)
        image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.enhancement_factor)
        image = np.array(image)
        image = grayscale_to_float(image)
        return image

# not used
class PCSCompressionTransform(PCSTransform):
    def __init__(self, quality):
        self.quality = quality

    def apply_coords(self, coords):
        return coords

    def apply_rotated_box(self, rotated_boxes):
        return rotated_boxes

    def apply_image(self, image):
        image = grayscale_to_uint(image)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode(".jpg", image, encode_param)
        io_buf = io.BytesIO(buffer)
        compressed_image = cv2.imdecode(
            np.frombuffer(io_buf.getbuffer(), np.uint8), cv2.IMREAD_GRAYSCALE
        )
        return grayscale_to_float(compressed_image)


class PCSBlurringTransform(PCSTransform):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def apply_image(self, image):
        image = grayscale_to_uint(image)
        image = cv2.GaussianBlur(
            image, (self.kernel_size, self.kernel_size), cv2.BORDER_REFLECT
        )
        return grayscale_to_float(image)


class PCSDistortionTransform(PCSTransform):
    def __init__(self, shift_lambda):
        self.shift_lambda = shift_lambda

    def apply_image(self, image):
        image = image.copy()
        for i in range(image.shape[0]):
            image[:, i] = np.roll(image[:, i], int(self.shift_lambda(i)))
            image[i, :] = np.roll(image[i, :], int(self.shift_lambda(i)))
        return image

class PCSFilterUndetectableTransform(PCSTransform):
    def __init__(self, num_objects, min_contrast_variation):
        self.num_objects = num_objects
        self.detectable_flags = [True for object in range(self.num_objects)]
        self.object_index = 0
        self.min_contrast_variation = min_contrast_variation

    def apply_image(self, image):
        self.image = image.copy()
        return image

    def apply_segmentation(self, segmentation):
        assert hasattr(
            self, "image"
        ), "Method apply_image needs to be called before apply_segmentation"
        mask = np.zeros(self.image.shape, dtype=np.uint8)
        int_segmentation = np.int64(np.around(segmentation))
        cv2.drawContours(mask, [int_segmentation], 0, 1, 4)
        cv2.drawContours(mask, [int_segmentation], 0, 1, -1)
        object_pixels = self.image[mask > 0]
        contrast = object_pixels.max() - object_pixels.min()
        if contrast <= self.min_contrast_variation:  # pixel values between [0, 1]
            self.detectable_flags[self.object_index] = False
        self.object_index += 1
        return segmentation


class PCSFilterInvalidTransform(PCSTransform):
    def __init__(self, num_objects, min_edge_length):
        self.num_objects = num_objects
        self.valid_flags = [True for object in range(self.num_objects)]
        self.object_index = 0
        self.min_edge_length = min_edge_length

    def apply_rotated_box(self, rotated_box):
        width = rotated_box[2]
        height = rotated_box[3]
        theta = rotated_box[4] + 90
        if width < self.min_edge_length or height < self.min_edge_length or width < height or theta < 0 or theta > 180:
            self.valid_flags[self.object_index] = False
        self.object_index += 1
        return rotated_box


class PCSFlipTransform(PCSTransform):
    def __init__(self, flip):
        self.flip = 0 if flip == "h" else 1

    def apply_coords(self, coords):
        coords[:, self.flip] = -(coords[:, self.flip] - self.origin) + self.origin
        return coords

    def apply_rotated_box(self, rotated_boxes):
        rotated_boxes = np.array(rotated_boxes)
        rotated_boxes[:2] = self.apply_coords(rotated_boxes[:2][np.newaxis, :])[0]
        rotated_boxes[4] = -rotated_boxes[4]
        return rotated_boxes.tolist()

    def apply_segmentation(self, segmentation):
        return self.apply_coords(segmentation)

    def apply_image(self, image):
        self.shape = image.shape
        self.origin = self.shape[self.flip] / 2
        if self.flip == 0:
            return np.fliplr(image)
        else:
            return np.flipud(image)


class PCSRotationTransform(PCSTransform):
    def __init__(self, theta_degree):
        assert theta_degree % 90 == 0
        self.k = theta_degree // 90
        self.theta_degree = theta_degree
        self.rotation_matrix = get_rotation_matrix(self.theta_degree)

    def apply_coords(self, coords):
        coords = coords - self.center
        coords = self.rotation_matrix.dot(coords.T).T
        return coords + self.center

    def apply_image(self, image):
        self.shape = image.shape
        self.center = np.array([self.shape[0] / 2, self.shape[1] / 2])
        return np.rot90(image, k=-self.k)

    def apply_rotated_box(self, rotated_boxes):
        rotated_boxes[:2] = self.apply_coords(np.array(rotated_boxes[:2]))
        rotated_boxes[4] = correct_angle(rotated_boxes[4] + self.theta_degree)
        return rotated_boxes

    def apply_segmentation(self, segmentation):
        return self.apply_coords(segmentation)

class PCSContrastTransform(PCSTransform):
    def __init__(self, enhancement_factor=1.0):
        self.enhancement_factor = enhancement_factor

    def apply_image(self, image):
        image = grayscale_to_uint(image)
        image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.enhancement_factor)
        image = np.array(image)
        image = grayscale_to_float(image)
        return image
