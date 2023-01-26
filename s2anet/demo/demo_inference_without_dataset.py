import argparse
import os
import os.path as osp
import pdb
import random

import cv2
import mmcv
from mmcv import Config

from mmdet.apis import init_detector, inference_detector
from mmdet.core import rotated_box_to_poly_single
from mmdet.datasets import build_dataset


def show_result_rbox(img,
                     detections,
                     class_names,
                     scale=1.0,
                     threshold=0.5,
                     colormap=None,
                     show_label=False):
    assert isinstance(class_names, (tuple, list))
    if colormap:
        assert len(class_names) == len(colormap)
    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for j, name in enumerate(class_names):
        if colormap:
            color = colormap[j]
        else:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        # import ipdb;ipdb.set_trace()
        for det in dets:
            score = det[-1]
            # print(score)
            det_bck = det.copy()
            det[-2] = -det[-2]
            det = rotated_box_to_poly_single(det[:-1])
            bbox = det[:8] * scale
            if score < threshold:
                continue
            # print(det_bck)
            bbox = list(map(int, bbox))

            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=color,
                         thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
            if show_label:
                cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img

import math
def preproc(img_path, tmp_img_path):
    imgdir = os.path.dirname(img_path)
    imgname = os.path.basename(img_path)
    nimg_path = os.path.join(imgdir, 'p' + imgname)
    img = cv2.imread(img_path)
    shape = img.shape
    if len(shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    r = w / h
    maxsize = 1900000
    f = math.sqrt(maxsize / (h * w))
    if f < 1:
        nw = int(w * f)
        nh = int(h * f)
        ctrx = w / 2
        ctry = h / 2
        img = img[int(ctry - nh / 2):int(ctry + nh / 2), int(ctrx - nw / 2):int(ctrx + nw / 2)]
    cv2.imwrite(tmp_img_path, img)
    return tmp_img_path

import tempfile
def save_det_result(config_file, out_dir, checkpoint_file=None, img_dir=None, colormap=None):
    cfg = Config.fromfile(config_file)
    data_test = cfg.data.test
    # dataset = build_dataset(data_test)
    classnames = ["Crystal",]
    # use checkpoint path in cfg
    if not checkpoint_file:
        checkpoint_file = osp.join(cfg.work_dir, 'latest.pth')
    # use testset in cfg
    if not img_dir:
        img_dir = data_test.img_prefix

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = osp.join(img_dir, img_name)
        img_out_path = osp.join(out_dir, img_name)
        pre_out_path = osp.join(out_dir, "p"+img_name)
        img_path = preproc(img_path, pre_out_path)
        result = inference_detector(model, img_path)
        img = show_result_rbox(img_path,
                            result,
                            classnames,
                            scale=1.0,
                            threshold=0.5,
                            colormap=colormap)
        cv2.imwrite(img_out_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('config_file', help='input config file')
    parser.add_argument('model', help='pretrain model')
    parser.add_argument('img_dir', help='img dir')
    parser.add_argument('out_dir', help='output dir')
    args = parser.parse_args()

    dota_colormap = [
        (54, 67, 244)]

    hrsc2016_colormap = [(212, 188, 0)]
    save_det_result(args.config_file, args.out_dir, checkpoint_file=args.model, img_dir=args.img_dir,
                    colormap=dota_colormap)
