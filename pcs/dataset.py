import datetime
import glob
import gzip
import json
import os
import pickle
import random
import time
from hashlib import sha256

import cv2
import numpy as np
from tqdm import tqdm
from pcs.augmentations import PCSDefaultAugmentor

from pcs.utils import (
    convert_min_area_rect,
    grayscale_to_float,
    grayscale_to_uint
)


class CocoExporter:
    def __init__(self, output_dir="", dataset_name=""):
        self.output_dir = output_dir
        self.dataset_name = dataset_name

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.dataset_dir = os.path.join(self.output_dir, self.dataset_name)
        if not os.path.isdir(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        self.number_images = 0
        self.number_annotations = 0
        date = str(datetime.date.today())
        self.coco_dataset = {
            "info": {
                "description": "Protein crystals in suspension (PCS) dataset for automated crystal detection",
                "url": "",
                "version": "1.0",
                "year": 2021,
                "contributor": "Daniel Bischoff, Sebastian Franz",
                "date_created": date,
            },
            "licenses": [
                {
                    "url": "https://opensource.org/licenses/MIT",
                    "id": 1,
                    "name": "MIT License",
                },
            ],
            "categories": [
                {"supercategory": "Crystal", "id": 1, "name": "Crystal"},
            ],
            "images": [],
            "annotations": [],
        }

        self.images_template = {
            "license": 1,
            "file_name": "",
            "width": -1,
            "height": -1,
            "date_captured": date,
            "id": 0,
        }

        self.annotations_template = {
            "segmentation": [
                []
            ],
            "area": 0,
            "iscrowd": 0,
            "image_id": 0,
            "bbox": [0, 0, 0, 0],
            "category_id": 1,
            "id": 0,
        }

    def add_image(self, image_path, height, width):
        self.number_images += 1
        image_id = self.number_images
        image_name = f"{str(image_id).zfill(10)}"
        _, ext = os.path.splitext(image_path)
        image_dict = self.images_template.copy()
        image_dict["file_name"] = image_name + ext
        image_dict["width"] = width
        image_dict["height"] = height
        image_dict["id"] = image_id
        self.coco_dataset["images"].append(image_dict)
        return image_dict


    def add_annotation(self, image_id=1, segmentation=None, bbox=None, area=0):
        self.number_annotations += 1
        annotation_id = self.number_annotations

        if segmentation is None:
            segmentation = [[]]
        if bbox is None:
            bbox = []

        # Annotation
        annotation_dict = self.annotations_template.copy()
        annotation_dict["segmentation"] = segmentation
        annotation_dict["bbox"] = bbox
        annotation_dict["image_id"] = image_id
        annotation_dict["id"] = annotation_id
        annotation_dict["area"] = area
        self.coco_dataset["annotations"].append(annotation_dict)
        return annotation_id

    def write(self):
        dataset_annotations_file = os.path.join(
            self.output_dir,
            self.dataset_name + ".json"
        )
        with open(dataset_annotations_file, "w") as f:
            json.dump(self.coco_dataset, f, indent=None)


class Indexer:

    def __init__(self, root_dirs, labels={"train": 80, "validation": 20}):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        for root_dir in root_dirs:
            assert os.path.isdir(root_dir), f"Not a directory: {root_dir}"
        self.root_dirs = root_dirs

        sum_weights = sum(labels.values())
        self.labels = {
            label: weight / sum_weights
            for label, weight in labels.items()
        }
        assert sum(self.labels.values()) == 1


    def _index_iopairs(self, reindex):
        iopairs = {}
        for root_dir in self.root_dirs:
            glob_str = os.path.join(root_dir, "**", "*.png")
            inputs = glob.glob(glob_str, recursive=True)
            for image_file in tqdm(
                inputs,
                desc=f"Indexing io pairs from directory {root_dir}",
                total=len(inputs)
            ):
                index_file = image_file + ".idx.json"
                annotation_file = image_file + ".json"

                if not reindex and os.path.exists(index_file):
                    with open(index_file, "r") as f:
                        d_iopair = json.load(f)
                else:

                    d_iopair = {}
                    d_iopair["valid"] = os.path.exists(annotation_file)
                    image_data = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                    d_iopair["height"], d_iopair["width"] = image_data.shape
                    d_iopair["key"] = sha256(image_data.data.tobytes()).hexdigest()
                    if d_iopair["key"] in iopairs:
                        d_iopair["valid"] = False
                        print(f"warning: invalidating {image_file} (duplicate)")
                    
                    # validate annotations
                    try:
                        with open(annotation_file, "r") as f:
                            annotations = json.load(f)
                    except json.JSONDecodeError:
                        d_iopair["valid"] = False
                        print(f"warning: invalidating {image_file} (JSON decode error)")
                    if not "segmentation" in annotations:
                        d_iopair["valid"] = False
                        print(f"warning: invalidating {image_file} (missing segmentation field)")


                    # shape check
                    arr = np.array(annotations["segmentation"])
                    if len(arr.shape) < 3:
                        d_iopair["valid"] = False
                        print(f"warning: invalidating {image_file} (wrong segmentation shape)")
                    if arr.shape[2] != 3:
                        d_iopair["valid"] = False
                        print(f"warning: invalidating {image_file} (wrong segmentation shape)")
                    

                    # coordinated check
                    y_min = arr[:, :, 1].min()
                    y_max = arr[:, :, 1].max()
                    if y_min < 0 or y_max >= d_iopair["height"]:
                        d_iopair["valid"] = False
                        print(f"warning: invalidating {image_file} (coordinate out of image bounds)")
                    x_min = arr[:, :, 1].min()
                    x_max = arr[:, :, 1].max()
                    if x_min < 0 or x_max >= d_iopair["width"]:
                        d_iopair["valid"] = False
                        print(f"warning: invalidating {image_file} (coordinate out of image bounds)")
                        

                    d_iopair["label"] = ""
                    with open(index_file, "w") as f:
                        json.dump(d_iopair, f)

                iopairs[d_iopair["key"]] = (
                    image_file,
                    annotation_file,
                    index_file,
                    d_iopair["height"],
                    d_iopair["width"],
                    d_iopair["label"],
                    d_iopair["valid"]
                )
        
        return iopairs

    def _load_only(self):
        iopairs = {}
        for root_dir in self.root_dirs:
            glob_str = os.path.join(root_dir, "**", "*.png")
            inputs = glob.glob(glob_str, recursive=True)
            for image_file in inputs:
                index_file = image_file + ".idx.json"
                annotation_file = image_file + ".json"

                with open(index_file, "r") as f:
                    d_iopair = json.load(f)

                iopairs[d_iopair["key"]] = (
                    image_file,
                    annotation_file,
                    index_file,
                    d_iopair["height"],
                    d_iopair["width"],
                    d_iopair["label"],
                    d_iopair["valid"]
                )

        return iopairs

    def _resample_iopairs(self, iopairs):

        keys = list(iopairs.keys())
        random.shuffle(keys)
        offset = 0
        for label, fraction in self.labels.items():
            size = int(round(fraction * len(iopairs)))
            label_keys = keys[offset:offset+size]
            offset += size
            for key in label_keys:
                _, _, index_file, height, width, _, valid = iopairs[key]
                d_iopair = {
                    "key": key,
                    "height": height,
                    "width": width,
                    "label": label,
                    "valid": valid
                }
                with open(index_file, "w") as f:
                    json.dump(d_iopair, f)

    def load_iopairs(self, reindex=False):
        iopairs = self._index_iopairs(reindex)
        filtered_iopairs = {key: iopair for key, iopair in iopairs.items() if iopair[6]}
        self._resample_iopairs(filtered_iopairs)
        updated_iopairs = self._load_only()
        label_count = {label: 0 for label, _ in self.labels.items()}
        for _, iopair in updated_iopairs.items():
            label_count[iopair[5]] += 1
        print("after indexing:")
        for root_dir in self.root_dirs:
            print(f"\t{root_dir}")
        for label, count in label_count.items():
            print(f"\t{label}: {count} ({round(100 * self.labels[label], 2)}%)")
        return updated_iopairs
    

    def to_coco(self, output_dir, iopairs=None, box_mode="xywha", reindex=False, flip_y=True,  digits=2):        
        assert box_mode in ("xywha", "coco")

        exporters = {
            label: CocoExporter(
                output_dir=output_dir,
                dataset_name=f"pcs_{label}")
            for label, _ in self.labels.items()
        }

        label_dirs = {
            label: os.path.join(output_dir, f"pcs_{label}")
            for label, _ in self.labels.items()
        }

        if iopairs is None:
            valid_labeled_iopairs = self.load_iopairs(reindex=reindex)
        else:
            valid_labeled_iopairs = iopairs

        for _, iopair in tqdm(
            valid_labeled_iopairs.items(),
            desc=f"Exporting dataset to coco format",
            total=len(valid_labeled_iopairs),
        ):
            image_file, annotation_file, _, height, width, label, _ = iopair
            exporter = exporters[label]

            # Adding image to dataset while ensuring that only grayscale images are stored
            image_dict = exporter.add_image(image_file, height, width)
            image_store_path = os.path.join(label_dirs[label], image_dict["file_name"])
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(image_store_path, image)

            # Adding annotations to image
            with open(annotation_file, "r") as f:
                annotations = json.load(f)

            segmentations = annotations["segmentation"]
            for verts in segmentations:
                # x, y, z device coordinates scaled and shifted to fit image dimesions of every crystal vertex
                verts = np.array(verts)
                assert verts.shape[-1] == 3 
                # depth information is currently not used.
                # the array is copied during np.delete which prevents a CV2 error.
                verts = np.delete(verts, 2, axis=1)

                if flip_y:
                    verts[:, 1] = (image_dict["height"] - 1) - verts[:, 1]

                # let CV2 figure out the correct ordering of the vertices
                hull = cv2.convexHull(np.float32(verts))

                # rounding to make the resulting JSON files smaller.
                area = round(cv2.contourArea(hull), digits)
                segmentation = [
                    [round(v, digits) for v in hull.flatten().tolist()]
                ]
                
                if box_mode == "coco":
                    x0 = verts[:, 0].min()
                    y0 = verts[:, 1].min()
                    w = verts[:, 0].max() - x0
                    h = verts[:, 1].max() - y0
                    bbox = [round(v, digits) for v in [x0, y0, w, h]]
                elif box_mode == "xywha":
                    min_area_rect = cv2.minAreaRect(hull)
                    bbox = convert_min_area_rect(min_area_rect)
                    bbox = [round(v, digits) for v in bbox]
                    
                exporter.add_annotation(
                    image_id=image_dict["id"],
                    segmentation=segmentation,
                    bbox=bbox,
                    area=area
                )

        for _, exporter in exporters.items():
            exporter.write()

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




class PCSDataset:

    def __init__(self, dataset_file, image_dir=None, use_augmentations=False, intermediates=False, periodic=False):
        assert os.path.exists(dataset_file)
        self.dataset_file = dataset_file
        assert self.dataset_file.lower().endswith(".json") or self.dataset_file.lower().endswith(".gzip")
        self.compressed = True if self.dataset_file.lower().endswith(".gzip") else False

        if not image_dir:
            dirname = os.path.dirname(self.dataset_file)
            basename = os.path.basename(self.dataset_file)
            name, _ = os.path.splitext(basename)
            if self.compressed and name.endswith(".json"): # support for .json.gzip
                name = name[:-len(".json")]
            image_dir = os.path.join(dirname, name)

        assert os.path.isdir(image_dir), f"Image directory not found: {image_dir}"
        self.image_dir = image_dir
        self.stats_file = self.image_dir + "_stats.json"
        self.aug_stats_file = self.image_dir + "_aug_stats.json"

        if self.compressed:
            print("Reading compressed PCSCocoDataset:", dataset_file, "...")
            start = time.time()
            with gzip.open(dataset_file, "r") as file:
                self.data = json.loads(file.read().decode('utf-8'))
            end = time.time()
            print("finished reading in", f"{round(end-start, 3)} seconds")

        else:
            print("Reading PCSCocoDataset:", dataset_file, "...")
            start = time.time()
            with open(dataset_file, "r") as file:
                self.data = json.load(file)
            end = time.time()
            print("finished reading in", f"{round(end-start, 3)} seconds")

        self.image_annotations = dict()
        for annotation in self.data["annotations"]:
            image_id = annotation["image_id"]
            if not image_id in self.image_annotations:
                self.image_annotations[image_id] = list()
            self.image_annotations[image_id].append(annotation)

        self.augmentations_active = use_augmentations
        self.intermediates = intermediates
        self.periodic = periodic
        self.augmentation_list = None

    def use_augmentations(self, augmentation_list=None, flag=True, intermediates=False):
        self.augmentations_active = flag
        self.intermediates = intermediates
        self.augmentation_list = augmentation_list

    def write_statistics(self, num_images=20000, digits=2):
        
        dataset_statistics = {
            "pixel_mean": -1,
            "pixel_std": -1,
            "num_images": -1,
            "num_annotations": -1,
            "augmentations_active": self.augmentations_active,
            "images": {
                "image_id": [],
                "height": [],
                "width": [],
                "instances_mean_area": [],
                "instances_mean_ratio": [],
                "annotation_ids": []
            },
            "annotations": {
                "image_id": [],
                "annotation_id": [],
                "x": [],
                "y": [],
                "width": [],
                "height": [],
                "angle": [],
                "area": [],
                "ratio": []
            }
        }

        dataset_statistics["num_images"] = len(self)
        num_annotations = 0

        image_stats_num_images = min(len(self), num_images)
        image_stats_indices = set(random.sample(range(image_stats_num_images), image_stats_num_images))
        image_flat_store = []

        image_stats = dataset_statistics["images"]
        annotation_stats = dataset_statistics["annotations"]

        def rndf(x):
            return round(float(x), digits)

        for index, img_data in enumerate(tqdm(
            self, total=len(self), desc="calculate image stats"
        )):
            if self.augmentations_active:
                image = img_data["aug_img"]
            else:
                image = img_data["img"]
            
            if index in image_stats_indices:
                image_flat_store.append(image.flatten().astype(np.float64))

            image_stats["image_id"].append(img_data["meta"]["img_dict"]["id"])
            image_shape = image.shape
            image_stats["height"].append(image_shape[0])
            image_stats["width"].append(image_shape[1])

            image_instance_areas  = []
            image_instance_ratios = []
            image_stats["annotation_ids"] = img_data["anno_ids"]

            if self.augmentations_active:
                segms = img_data["aug_segms"]
                bboxs = img_data["aug_rbboxs"]
            else:
                segms = img_data["segms"]
                bboxs = img_data["rbboxs"]

            for segmentation, rotated_box in zip(segms, bboxs):

                num_annotations += 1

                annotation_stats["image_id"].append(
                    img_data["meta"]["img_dict"]["id"]
                )

                x_ctr, y_ctr, width, height, angle = rotated_box
                annotation_stats["x"].append(rndf(x_ctr))
                annotation_stats["y"].append(rndf(y_ctr))
                annotation_stats["width"].append(rndf(width))
                annotation_stats["height"].append(rndf(height))
                annotation_stats["angle"].append(rndf(angle))

                ratio = width / (height + 1e-4)
                image_instance_ratios.append(rndf(ratio))
                annotation_stats["ratio"].append(rndf(ratio))

                area = cv2.contourArea(np.float32(segmentation))
                image_instance_areas.append(rndf(area))
                annotation_stats["area"].append(rndf(area))
            
            image_stats["instances_mean_area"].append(
                rndf(np.mean(image_instance_areas))
            )
            image_stats["instances_mean_ratio"].append(
                rndf(np.mean(image_instance_ratios))
            )

        image_flat_store = np.concatenate(image_flat_store)
        dataset_statistics["pixel_mean"] = rndf(np.mean(image_flat_store))
        dataset_statistics["pixel_std"] = rndf(np.std(image_flat_store))
        dataset_statistics["num_annotations"] = num_annotations

        output_file = self.aug_stats_file if self.augmentations_active else self.stats_file
        with open(output_file, "w") as f:
            json.dump(dataset_statistics, f)


    def write_augmented_dataset(self, output_dataset, digits=2):
        dirname = os.path.dirname(self.dataset_file)
        coco_exporter = CocoExporter(
            output_dir=dirname, dataset_name=output_dataset
        )
        self.use_augmentations()
        def rndf(x):
            return round(float(x), digits)
        for img_data in tqdm(
            self, total=len(self), desc="augmenting dataset"
        ):
            img_path = img_data["meta"]["img_path"]
            height, width = img_data["img"].shape
            image_dict = coco_exporter.add_image(
                img_path, int(height), int(width)
            )
            cv2.imwrite(
                os.path.join(
                    dirname,
                    output_dataset,
                    img_data["meta"]["img_dict"]["file_name"]
                ),
                grayscale_to_uint(img_data["aug_img"])
            )
            image_id = image_dict["id"]
            for segmentation, rotated_box in zip(img_data["aug_segms"], img_data["aug_rbboxs"]):
                area = rndf(cv2.contourArea(np.float32(segmentation)))
                segmentation = segmentation.flatten().tolist()
                segmentation = [rndf(v) for v in segmentation]
                if isinstance(rotated_box, np.ndarray):
                    rotated_box = rotated_box.flatten().tolist()
                    rotated_box = [rndf(v) for v in rotated_box]
                coco_exporter.add_annotation(
                    image_id=image_id,
                    segmentation=[segmentation],
                    bbox=rotated_box,
                    area=area
                )
                
        coco_exporter.write()


    def write_trimmed_dataset(self, output_dataset, digits=2, num=20, augmented=False):
        dirname = os.path.dirname(self.dataset_file)
        coco_exporter = CocoExporter(
            output_dir=dirname, dataset_name=output_dataset
        )
        if augmented:
            self.use_augmentations()
        def rndf(x):
            return round(float(x), digits)
        for idx, img_data in enumerate(tqdm(
            self, total=num, desc="trimming dataset"
        )):
            if idx == num:
                break
            img_path = img_data["meta"]["img_path"]
            height, width = img_data["img"].shape
            image_dict = coco_exporter.add_image(
                img_path, int(height), int(width)
            )
            cv2.imwrite(
                os.path.join(
                    dirname,
                    output_dataset,
                    img_data["meta"]["img_dict"]["file_name"]
                ),
                grayscale_to_uint(img_data["aug_img" if augmented else "img"])
            )
            image_id = image_dict["id"]
            for segmentation, rotated_box in zip(img_data["aug_segms" if augmented else "segms"], img_data["aug_rbboxs" if augmented else "rbboxs"]):
                area = rndf(cv2.contourArea(np.float32(segmentation)))
                segmentation = segmentation.flatten().tolist()
                segmentation = [rndf(v) for v in segmentation]
                if isinstance(rotated_box, np.ndarray):
                    rotated_box = rotated_box.flatten().tolist()
                    rotated_box = [rndf(v) for v in rotated_box]
                coco_exporter.add_annotation(
                    image_id=image_id,
                    segmentation=[segmentation],
                    bbox=rotated_box,
                    area=area
                )
                
        coco_exporter.write()


    def write_pickled_dataset(self, output_dataset):
        dirname = os.path.dirname(self.dataset_file)
        outpath = os.path.join(dirname, output_dataset)
        assert outpath.endswith(".pkl")
        data = []
        for idx, img_data in enumerate(tqdm(
            self, total=len(self), desc="writing segmented dataset"
        )):
            _, _, _, segmentations, rotated_boxes = img_data["meta"]["img_dict"], img_data["img"], None, img_data["segms"], img_data["rbboxs"]
            segmentations=[np.float32(segm) for segm in segmentations],
            rotated_boxes=np.float32(rotated_boxes)
            data.append((segmentations, rotated_boxes))
        with open(outpath, "wb") as f:
            pickle.dump(data, f)
        

    def load_aug_stats(self):
        with open(self.aug_stats_file, "r") as f:
            aug_stats = json.load(f)
            return aug_stats

    def load_stats(self):
        with open(self.stats_file, "r") as f:
            aug_stats = json.load(f)
            return aug_stats

    @staticmethod
    def get_segmentations(image_annotations):
        return [
            np.array(annotation["segmentation"], dtype=np.float32).flatten().reshape(-1, 2)
            for annotation in image_annotations
        ]

    @staticmethod
    def get_rotated_boxes(image_annotations, segmentations):
        # use bbox field if angle information is present, otherwise infer from segmentations
        assert len(image_annotations) > 0
        has_angle = len(image_annotations[0]["bbox"]) == 5
        if has_angle:
            return [
                np.array(annotation["bbox"], dtype=np.float32).flatten()
                for annotation in image_annotations
            ]
        else:
            min_area_rects = [
                cv2.minAreaRect(segmentation)
                for segmentation in segmentations
            ]
            return [
                np.array(convert_min_area_rect(min_area_rect), dtype=np.float32)
                for min_area_rect in min_area_rects
            ]

    @staticmethod
    def get_annotation_ids(image_annotations):
        return [annotation["id"] for annotation in image_annotations]

    def get_meta(self, idx):
        image_dict = self.data["images"][idx]
        image_annotations = self.image_annotations[image_dict["id"]]
        image_path = os.path.join(self.image_dir, image_dict["file_name"])
        assert os.path.exists(image_path)
        return dict(
            img_dict=image_dict,
            img_path=image_path,
            img_annos=image_annotations
        )

    def __getitem__(self, idx):
        
        meta = self.get_meta(idx)
        image = grayscale_to_float(
            cv2.imread(meta["img_path"], cv2.IMREAD_GRAYSCALE)
        )

        
        segmentations = PCSDataset.get_segmentations(meta["img_annos"])
        rotated_boxes = PCSDataset.get_rotated_boxes(meta["img_annos"], segmentations)
        annotation_ids = PCSDataset.get_annotation_ids(meta["img_annos"])
        
        if self.periodic:
            image, segmentations, rotated_boxes = _make_periodic(image, segmentations, rotated_boxes)

        return dict(
            meta=meta,
            img=image,
            anno_ids=annotation_ids,
            segms=segmentations,
            rbboxs=rotated_boxes
        )

    def __iter__(self):
        if self.augmentations_active:
            return PCSDatasetAugmentedIterator(self)
        else:
            return PCSDatasetIterator(self)

    def __len__(self):
        return len(self.data["images"])


class PCSDatasetIterator:
    def __init__(self, pcs_coco_dataset):
        self.dataset = pcs_coco_dataset
        self.num_images = len(self.dataset.data["images"])
        self.index = 0

    def __next__(self):
        if self.index < self.num_images:
            img_data = self.dataset[self.index]
            self.index += 1
            return img_data
        else:
            raise StopIteration


class PCSDatasetAugmentedIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_iter = PCSDatasetIterator(
            self.dataset
        )
        self.augmentor = PCSDefaultAugmentor(augmentation_list=self.dataset.augmentation_list)

    def __next__(self):

        img_data = next(
            self.dataset_iter
        )

        aug_result = self.augmentor(
            img_data["img"].copy(),
            [x.copy() for x in img_data["segms"]],
            [x.copy() for x in img_data["rbboxs"]]
        )

        img_data.update(aug_result)

        return img_data

    def __iter__(self):
        return self