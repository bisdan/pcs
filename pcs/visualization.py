import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
from vcw.vcw_utils import ensure_image_rgb, get_corners, get_rotation_matrix


class PCSDrawer:

    def __init__(self, color=(0, 255, 0), thickness=1, threshold=0.5):

        self.color = color
        self.thickness = thickness

        self.draw_rotated_boxes_enabled = True
        self.rotated_boxes_color = (0, 255, 0)
        self.rotated_boxes_thickness = 1

        self.draw_segmentations_enabled = True
        self.segmentations_color = (0, 200, 0)
        self.segmentations_thickness = -1

        self.draw_scores_enabled = False
        self.rotated_boxes_f = 1

        self.threshold = threshold
        self.autocolor = True
        cmap = list(cm.get_cmap("Dark2").colors)[:-1]
        self.autocolors = []
        for color in cmap:
            r = int(round(255 * color[0]))
            g = int(round(255 * color[1]))
            b = int(round(255 * color[2]))
            self.autocolors.append((r, g, b,))


    def enable_rotated_boxes(self, enable=True, color=(0, 255, 0), thickness=1, f=1):
        self.draw_rotated_boxes_enabled = enable
        self.rotated_boxes_color = color
        self.rotated_boxes_thickness = thickness
        self.rotated_boxes_f = f

    def enable_segmentations(self, enable=True, color=(0, 255, 0), thickness=1):
        self.draw_segmentations_enabled = enable
        self.segmentations_color = color
        self.segmentations_thickness = thickness

    def enable_scores(self, enable=True):
        self.draw_scores_enabled = enable


    def draw_segmentations(self, image, segmentations=None, thickness=1, color_indices=None, color_weight=0.2, f=1, bth=0.2):

        if segmentations is not None:
            segmentations = [
                np.float64(np.around(contour)) for contour in segmentations
            ]
            for segm, cidx in zip(segmentations, color_indices):
                if thickness == -1:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    segm_mean = segm.mean(axis=0)
                    segm -= segm_mean
                    segm *= f
                    segm += segm_mean
                    segm = np.around(segm).astype(np.int32)
                    
                    mask = cv2.drawContours(mask, [segm], 0, 1, -1)
                    mask_inds = mask > 0
                    image[mask_inds] = np.uint8(
                        np.around(
                            (1 - color_weight) * np.float32(image[mask_inds]) + color_weight * np.array(self.autocolors[cidx%len(self.autocolors)])
                        )
                    )
                else:
                    segm = segm.astype(np.float32)
                    segm_mean = segm.mean(axis=0)
                    segm -= segm_mean
                    segm *= f
                    segm += segm_mean
                    segm = np.around(segm).astype(np.int32)
                    cv2.drawContours(
                        image,
                        [segm],
                        0,
                        self.autocolors[cidx%len(self.autocolors)],
                        thickness
                    )

    def apply_threshold(self, rotated_boxes, scores):
        assert scores is not None
        assert rotated_boxes is not None
        rotated_boxes = [rotated_box for score, rotated_box in zip(scores, rotated_boxes) if score >= self.threshold]
        scores = [score for score in scores if score >= self.threshold]
        return rotated_boxes, scores

    def approximate_valid_pixels(self, image):
        ret, thr = cv2.threshold(np.around(image.mean(axis=2)).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        return thr

    def get_autocoloring(self, image, rotated_boxes):
        shape = image.shape
        corners = [get_corners(rotated_box, f=-1.0) for rotated_box in rotated_boxes]
        global_mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        color_indices = []
        for box_corners in corners:
            mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
            box_corners = np.int32(np.around(box_corners))
            cv2.drawContours(mask, [box_corners], 0, 1, -1)
            mask_indices = np.around(mask) > 0
            color_index = np.max(global_mask[mask_indices])
            global_mask[mask_indices] += 1
            color_indices.append(color_index)
        return color_indices
    
    def __call__(
        self, 
        image,
        rotated_boxes=None,
        segmentations=None,
        scores=None
    ):
        canvas_image = ensure_image_rgb(image)
        input_image = canvas_image.copy()

        if self.autocolor:
            assert rotated_boxes is not None
            color_indices = self.get_autocoloring(input_image, rotated_boxes)
        else:
            color_indices = [0 for box in rotated_boxes]

        # draw segmentations
        if self.draw_segmentations_enabled:
            assert segmentations is not None
            self.draw_segmentations(
                canvas_image,
                segmentations,
                thickness=self.segmentations_thickness,
                color_indices=color_indices
            )

        # draw rotated boxes
        if self.draw_rotated_boxes_enabled:
            assert rotated_boxes is not None
            corners = [
                get_corners(rotated_box, f=-1.0) for rotated_box in rotated_boxes
            ]
            self.draw_segmentations(
                canvas_image,
                corners,
                thickness=self.rotated_boxes_thickness,
                color_indices=color_indices,
                f=self.rotated_boxes_f
            )

        if self.draw_scores_enabled:
            assert scores is not None
            for box, score, cidx in zip(rotated_boxes, scores, color_indices):
                x, y, w, h, theta = box
                dx = - w / 2
                dy = - h / 2
                mat = get_rotation_matrix(theta)
                dvec = np.dot(mat, np.array((dy, dx,)))
                y, x = np.array((y, x,)) + dvec
                pscore = int(round(100 * score))
                score_str = str(pscore)
                fontScale = (np.log2(w*h) + 8) / 12
                cv2.putText(canvas_image, score_str, np.int32(np.around(np.array((x, y,)))), color=self.autocolors[cidx%len(self.autocolors)], fontScale=fontScale, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    
        return input_image, canvas_image

    def draw_grid(self, dataset, output_dir=None, grid=(4, 4), figsize=None, num_images=10, ids=None):
        if output_dir and not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        images_per_grid = grid[0] * grid[1]
        images = []
        num_drawn = 0
        for image_dict, image, _, segmentations, rotated_boxes in tqdm(
            dataset, total=len(dataset), desc="drawing rotated boxes"
        ):
            id = int(image_dict["id"])
            if ids is not None and not id in ids:
                continue

            _, image = self(image, rotated_boxes, segmentations)
            images.append(image)
            if len(images) == images_per_grid:
                if figsize is not None:
                    fig = plt.figure(figsize=figsize)
                else:
                    fig = plt.figure()
                image_grid = ImageGrid(fig, 111, nrows_ncols=grid, axes_pad=0.1)
                for ax, img in zip(image_grid, images):
                    ax.imshow(img)
                    ax.axis("off")
                plt.tight_layout()
                if output_dir:
                    plt.savefig(os.path.join(output_dir, image_dict["file_name"]))
                else:
                    plt.show()
                num_drawn += 1
                if num_drawn >= num_images:
                    break
                images = []

