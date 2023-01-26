import os.path as osp

import mmcv
import json
import math
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .pipelines.transforms import Pad
from .pipelines.transforms_rotated import RotatedResize, RotatedRandomFlip, RandomRotate
from .pipelines.formating import DefaultFormatBundle, Collect
from .registry import DATASETS

import pickle
import cv2
import random
import matplotlib.pyplot as plt
from pcs.augmentations import PCSDefaultAugmentor, PCSMinimalAugmentationList, PCSContrastAugmentationList, PCSPerlinAugmentationList, PCSWaveAugmentationList
from pcs.visualization import PCSDrawer

def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]

# def norm_angle(angle, range=[-np.pi / 2.0, np.pi]):
#     return (angle - range[0]) % range[1] + range[0]


import itertools
def _make_periodic(img, segms, bboxs):
    """Make image periodic by adding a copy of the image to the right and bottom of the image.
    The segmentation and bounding boxes are adjusted accordingly.
    """
    img = img[:-2, :-2] 
    imgs = [img, np.flip(img, axis=0), np.flip(img, axis=1), np.flip(img, axis=(0, 1))]
    
    nimg = np.empty((2 * img.shape[0], 2 * img.shape[1]), dtype=img.dtype)
    nsegms = []
    nbboxs = []
    for i, j in itertools.product(range(2), range(2)):
        xo = j * img.shape[0]
        yo = i * img.shape[1]
        nimg[xo:xo + img.shape[0], yo:yo + img.shape[1]] = imgs[i * 2 + j]
        if i == 0 and j == 0:
            fang = 1
            fdx = 0
            fdy = 0
        elif i == 0 and j == 1:
            fang = -1
            fdx = 1 
            fdy = 0
        elif i == 1 and j == 0:
            fang = -1
            fdx = 0
            fdy = 1
        else:
            fang = 1
            fdx = 1 
            fdy = 1
        for segm in segms:
            seg = segm.copy()
            seg[:, 0] += yo - 2 * fdy *(seg[:, 0] - img.shape[1]/2.0)
            seg[:, 1] += xo - 2 * fdx * (seg[:, 1] - img.shape[0]/2.0)
            nsegms.append(seg)
        for box in bboxs:
            nbboxs.append(
                np.array(
                    [
                        box[0] + xo - 2 * fdx * (box[0] - img.shape[0]/2.0),
                        box[1] + yo - 2 * fdy *(box[1] - img.shape[1]/2.0),
                        box[2], box[3], box[4]*fang
                    ]
                )
            )

    return nimg, nsegms, nbboxs
    # return _make_periodic(nimg, nsegms, nbboxs) if nimg.shape[0] < 1024 and nimg.shape[1] < 1024 else (nimg, nsegms, nbboxs)


@DATASETS.register_module
class PCSTrainDataset:
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ('crystal',)

    

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 img_mean=127.5,
                 img_std=73.95,
                 augmentation_list=PCSMinimalAugmentationList):
        super().__init__()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.img_mean = augmentation_list.img_mean
        self.img_std = augmentation_list.img_std
        self.augmentor = PCSDefaultAugmentor(augmentation_list=augmentation_list)
        # self.drawer = PCSDrawer()

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        # load annotations
        # self.img_infos = self.load_annotations(self.ann_file)
        with open(ann_file, "rb") as f:
            self.data = pickle.load(f)


        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def get_img_info(self, idx):
        info = dict()
        img_name = f"{str(idx+1).zfill(10)}.png"
        segm, rotated_boxes = self.data[idx]
        segm = np.array(segm[0])
        info = {
            'filename': osp.join(self.img_prefix, img_name),
            'width': 384,
            'height': 384,
            'ann': {
                'bboxes': rotated_boxes,
                'segm': segm,
                'labels': [1 for _ in rotated_boxes]
            }
        }
        return info
            
    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def prepare_train_img(self, idx):
        img_info = self.get_img_info(idx)
        ann_info = img_info["ann"]
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return results

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.prepare_train_img(idx)

        image_path = data["img_info"]["filename"]
        ann_info = data["ann_info"]
        data["filename"] = image_path
        assert osp.exists(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        bboxes = ann_info["bboxes"]
        segms = ann_info["segm"]
        # image, segms, bboxes = _make_periodic(image, segms, bboxes)
        labels = [1 for bbox in bboxes]
        while True:
            aug_result = self.augmentor(
                image.copy(),
                [x.copy() for x in segms],
                [x.copy() for x in bboxes]
            )
            aug_image = aug_result["aug_img"]
            aug_bboxes = aug_result["aug_rbboxs"]
            aug_segms = aug_result["aug_segms"]
            if len(aug_bboxes) > 0:
                break
            
        
        for box in aug_bboxes:
            box[-1] = norm_angle(math.radians(box[-1]))

        img_mean = aug_image.mean()
        img_std = aug_image.std()
        # print(img_mean, img_std)
        aug_image -= img_mean
        # aug_image /= img_std
        data["img_norm_cfg"] = dict(
            mean=img_mean,
            std=img_std,
            to_rgb=False
        )
        data["img"] = aug_image.astype(np.float32)
        data["ori_shape"] = (*image.shape, 1)
        data["img_shape"] = (*image.shape, 1)
        data["pad_shape"] = (*image.shape, 1)
        data["scale_factor"] = 1.0
        data["gt_bboxes"] = np.array(aug_bboxes, dtype=np.float32)
        data["gt_labels"] = np.array(labels, dtype=np.int64)
        data["img"] = data["img"][:, :, np.newaxis]
        data["flip"] = False
        data['bbox_fields'] = ['gt_bboxes']

        # rot_res = RotatedResize(img_scale=[(384, 384),], keep_ratio=True, multiscale_mode="value")
        # data = rot_res(data)
        # if len(data["img"].shape) == 2:
        #     data["img"] = data["img"][:, :, np.newaxis]
        # if len(data["img_shape"]) == 2:
        #     data["img_shape"] = (*data["img_shape"], 1)
        # rot_rand_flip = RotatedRandomFlip(flip_ratio=0.5)
        # data = rot_rand_flip(data)
        # if len(data["img"].shape) == 2:
        #     data["img"] = data["img"][:, :, np.newaxis]
        # if len(data["img_shape"]) == 2:
        #     data["img_shape"] = (*data["img_shape"], 1)
        # rand_rot = RandomRotate(rate=0.5, angles=[30, 60, 90, 120, 150], auto_bound=False)
        # data = rand_rot(data)
        # if len(data["img"].shape) == 2:
        #     data["img"] = data["img"][:, :, np.newaxis]
        # if len(data["img_shape"]) == 2:
        #     data["img_shape"] = (*data["img_shape"], 1)

        # print("a", data["gt_bboxes"])
        pad = Pad(size_divisor=32)
        dfb = DefaultFormatBundle()
        collect = Collect(keys=['img', 'gt_bboxes', 'gt_labels'])
        data = pad(data)
        data = dfb(data)
        # # print("b", data["gt_bboxes"])
        data = collect(data)
        # print("c", data["gt_bboxes"])
        return data

