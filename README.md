# Virtual Protein crystals in suspension (PCS)

Generating large-scale synthetic data sets of crystallization processes for supervised machine-learning.

## PCS data sets
The datasets are distributed in the COCO format and can be downloaded from [LRZ Sync+Share](https://syncandshare.lrz.de/getlink/fiQmpeVNi4XKJ9ioGLsTSVJY).
- **pcs_train** (322558 images, 18.1 GiB): training set
- **pcs_validation** (10000 images, 569.8 MiB): validation set (not augmented)
- **pcs_validation_aug** (10000 images, 771.1 MiB): augmented validation set


## Installation
If you want to use our data loader for machine-learning and/or experiment with augmentations:
```bash
pip3 install -e .
```
However, if you require a customized data set and can not achieve the desired visual effect through customized data augmentations continue to the [Blender Add-on](#pcs-blender-add-on) section.

### Data augmentation and examples
The PCS project handles data augmentations similar to Detectron2. The randomness of data augmentation is handled by PCSAugmentation objects. These create PCSTransform objects that apply the specific effect to the input images. To use the default augmentation list (PCSDefaultAugmentor):

```python
from pcs.dataset import PCSDataset

pcs_coco = PCSDataset("pcs_validation.json") # .json needs to be in the same directory as the pcs_validation image dir
pcs_coco.use_augmentations() # use default augmentation pipeline
for img_info in pcs_coco:
    meta = img_info["meta"]
    base_img = img_info["img"]
    annotation_ids = img_info["anno_ids"]
    segmentations = img_info["segms"]
    rotated_bboxs = img_info["rbboxs"]
    augmented_img = img_info["aug_img"]
    augmented_segmentations = img_info["aug_segms"]
    augmented_rotated_bboxs = img_info["aug_rbboxs"]
    # ...
    continue
```
or alternatively via indexing (e.g. required by the PyTorch data loader)
```python
from pcs.dataset import PCSDataset
import random
pcs_coco = PCSDataset("pcs_validation.json") 
pcs_coco.use_augmentations()
idx = random.randrange(len(pcs_coco))
img_info = pcs_coco[idx]
```
For quick visualizations of the augmented images and annotations:
```python
from pcs.dataset import PCSDataset
from pcs.visualization import PCSDrawer
import numpy as np
import matplotlib.pyplot as plt

pcs_coco = PCSDataset("pcs_validation.json") 
drawer = PCSDrawer()
pcs_coco.use_augmentations() 
for img_info in pcs_coco:
    base_img = img_info["img"]
    segmentations = img_info["segms"]
    rotated_bboxs = img_info["rbboxs"]
    augmented_img = img_info["aug_img"]
    augmented_segmentations = img_info["aug_segms"]
    augmented_rotated_bboxs = img_info["aug_rbboxs"]
    base_img, base_img_labels = drawer(
        base_img,
        rotated_bboxs,
        segmentations=segmentations
    )
    augmented_img, augmented_img_labels = drawer(
        augmented_img,
        augmented_rotated_bboxs,
        segmentations=augmented_segmentations
    )
    display_img = np.hstack(
        (base_img, base_img_labels, augmented_img, augmented_img_labels)
    )
    plt.imshow(display_img)
    plt.show()
```
- Without the use of augmentations (coloration is used to disambiguate overlapping crystals):
![](examples/gallery.png?raw=true)
![](examples/gallery_anno.png?raw=true)

- Using the default PCS augmentation pipeline (coloration is used to disambiguate overlapping crystals):
![](examples/gallery_aug.png?raw=true)
![](examples/gallery_aug_anno.png?raw=true)


## PCS Blender Add-on
### Installation instructions
1. Download and install [Blender](https://www.blender.org/download/) (tested: v2.93, v3.0).

2. The Blender add-on requires OpenCV python bindings `opencv-python`. 
Since Blender uses its own python version, users need to install the the `opencv-python` package 
with this specific python binary. 
This requires you to first locate blenders installation directory `BLENDER_DIR` and then execute the following commands.
```bash
cd ${BLENDER_DIR}/3.0/python/bin
./python3.9 -m ensurepip --user
./python3.9 -m pip install --upgrade pip
./python3.9 -m pip install opencv-python scipy
```
3. In blender, open *Edit -> Preferences -> File Paths*

4. Set scripts path to pcs/blender_addon

5. Restart blender

6. Go to *Edit -> Preferences -> Addons -> Search "Crystal Well"*

7. Enable the addon, it should stay enabled now at every start-up

8. Make sure that "Object Mode" is enabled. Select *General*, *3D Viewport (Press **N**) -> Crystal Well*


### Blender Headless usage instructions
If you have saved a settings_file.json in the blender addon UI, you can use this file for headless execution.
Simply call 
```bash
python3 headless_execution.py pcs_settings.json
```
**You might need to change the path of the blender executable in headless_execution.py**

Feel free to change the settings_file itself but make sure to use values that are reasonable. 
For example, it does make sense to change the output_path in the pcs_settings.json if you want to render a new set of images.

**Note:** The default settings require up to a few minutes for a single image. The PCS data sets were created by running multiple worker processes on many machines.
### Exporting the Output of the VCW Blender Add-on to COCO Format.
In the following example ```root_dir``` can contain multiple directories with input-output pairs from multiple Blender add-on workers. These will be indexed, validated, shuffled and then split into train and validation data sets. Images are copied into subdirectories of ```output_dir``` during this process.
```python
from pcs.dataset import Indexer

# Generate two datasets (80% and 20%) from the images contained in 'root_dir'
idx = Indexer("/path/to/root/dir", labels={"train": 80, "validation": 20})

# Find and label image-annotation pairs
# Then validate them and extract some additional information
iopairs = idx.load_iopairs()

# Export to COCO format
# Note that we use the box mode XYWHA
# We also round floats to two digits in the generated JSON file
idx.to_coco("path/to/output/dir", box_mode="xywha", digits=2)
```
The PCS data set is distributed in COCO format. If another format is required, then a corresponding export function needs to be implemented. However, one might still find the ```load_iopairs``` function of the Indexer useful.

