from s2anet.demo.demo_inference_without_dataset import save_det_result as _save_det_result
import os.path as osp
import tempfile
import glob
import cv2
import math
import s2anet.mmcv as mmcv
import numpy as np
import pandas as pd
from s2anet.mmdet.apis import init_detector, inference_detector
from pcs.visualization import PCSDrawer
import matplotlib.pyplot as plt

_model_name = "gn_s2a"
_models_dir = osp.join("..", "models")
_model_config_path=osp.join(_models_dir, _model_name + ".py")
_model_config=mmcv.Config.fromfile(_model_config_path)
_work_dirs = osp.join("..", "work_dirs")
_model_checkpoint=osp.join(_work_dirs, _model_name, "latest.pth")
_ai_model_checkpoint_origin = f"ai:repos/s2anet/work_dirs/{_model_name}/latest.pth"
_fetch_cmd = f"scp {_ai_model_checkpoint_origin} work_dirs/."
_tmpfs = ".tmpfs"
_fetch_cmd

def collect_input(input_path: str):
    if not osp.isdir(input_path) and not osp.isfile(input_path):
        raise ValueError("Path '%s' not found" % input_path)
    image_paths = list()
    if osp.isdir(input_path):
        glob_pattern = osp.join(input_path, "**/*.png")
        image_paths.extend(glob.glob(glob_pattern, recursive=True))
    elif osp.isfile(input_path):
        if not input_path.endswith(".png"):
            raise ValueError("Input path must be a png file")
        image_paths.append(input_path)
    if len(image_paths) == 0:
        raise ValueError("No images found in '%s'" % input_path)
    return sorted(image_paths)

def post_inference_hook(df, oimg, limg):
    img = np.hstack((oimg, limg))
    plt.imshow(img)
    plt.show()
    print(df.describe())


def inference(input_path: str, output_path: str = None, test_cfg: dict = None, test_pipeline: list = None):
    image_paths = collect_input(input_path)
    print("Loading model from '%s'" % _model_checkpoint)
    if test_cfg is None:
        test_cfg = _model_config["test_cfg"]
    if test_pipeline is None:
        test_pipeline = _model_config["test_pipeline"]
    model_config = _model_config._cfg_dict.to_dict().copy()
    model_config["test_cfg"] = test_cfg
    model_config["test_pipeline"] = test_pipeline
    model_config["data"]["test"]["pipeline"] = test_pipeline
    model_config = mmcv.Config(model_config)
    model = init_detector(model_config, _model_checkpoint, device='cuda:0')
    print("Inference on %d images" % len(image_paths))
    results = dict()
    for image_path in image_paths:
        print("Inference on '%s'" % image_path)
        assert not image_path in results
        df, oimg, limg = inference_single(model, image_path)
        results[image_path] = df
        post_inference_hook(df, oimg, limg)


def preprocess_single(image_path: str):
    if not osp.isfile(image_path):
        raise ValueError("Image path '%s' not found" % image_path)
    img = cv2.imread(image_path)
    # convert to grayscale if necessary
    shape = img.shape
    if len(shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    r = w / h
    maxsize = 1900000 # size of image is GPU memory limited
    f = math.sqrt(maxsize / (h * w))
    if f < 1:
        nw = int(w * f)
        nh = int(h * f)
        ctrx = w / 2
        ctry = h / 2
        img = img[int(ctry - nh / 2):int(ctry + nh / 2), int(ctrx - nw / 2):int(ctrx + nw / 2)]
    return img

def inference_single(model, image_path: str):
    with tempfile.NamedTemporaryFile(dir=_tmpfs, suffix=".png") as tmpf:
        cv2.imwrite(
            tmpf.name,
            preprocess_single(image_path)
        )
        result = inference_detector(model, tmpf.name)
        if not result:
            print("No crystal detected in '%s'" % image_path)
            return
        result = result[0]
        if result.shape[0] == 0:
            print("No crystal detected in '%s'" % image_path)
            return
        thr_crystals = list()
        widths = list()
        heights = list()
        areas = list()
        ratios = list()
        angles = list()
        scores = list()
        for i, res in enumerate(result):
            if res[-1] > threshold:
                res[-2] = math.degrees(res[-2])
                width = res[2]
                height= res[3]
                area = width * height
                ratio = width / (height + 1e-6)
                angle = res[-2]
                score = res[-1]
                widths.append(width)
                heights.append(height)
                areas.append(area)
                ratios.append(ratio)
                angles.append(angle)
                scores.append(score)
                thr_crystals.append(res)
        df = pd.DataFrame({
            "width": widths,
            "height": heights,
            "area": areas,
            "ratio": ratios,
            "angle": angles,
            "score": scores
        })
        drawer = PCSDrawer()
        drawer.draw_segmentations_enabled = False
        oimg, limg = drawer(
            img,
            thr_crystals
        )
        return df, oimg, limg
