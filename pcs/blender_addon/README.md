# PCS Blender Add-On

## Installation

Please refer to the main [README](https://github.com/bisdan/pcs/blob/main/README.md#pcs-blender-add-on) 
for installation instructions.

## How-To

After the installation, the add-on is accessible in object mode by
pressing "N" or opening the 3D-viewport sidebar. Here you should find the entry "Crystal Well".
In this panel, properties of the PCS data sets can be modified. Hovering each property displays additional details.

### Hints & Tipps
* The *Update UI* option lets you take a look at the resulting virtual scene after every render.
* If you want to use a *default import or output directory*, you can change the default values in 
modules/blvcw/crystal_well_global_state.py.

### Headless execution

Please refer to the main [README](https://github.com/bisdan/pcs/blob/main/README.md#pcs-blender-add-on) 
for headless execution instructions.

*Note: The default settings require up to a few minutes for a single image. 
The PCS data sets were created by running multiple worker processes on many machines.*

## Troubleshooting
Some combinations of settings might cause problems for the add-on (e.g. res_x >> res_y).
If you encounter such a situation, feel free to let us know and create an issue!

## Architecture of the add-on
The `CrystalWellLoader` component was created to manage import and loading of single crystals.
The user interface manages and controls all settings. When the user hits the "Render" button, all settings are saved in the `CrystalWellSettings` component. 
After a CrystalWellSimulator component is created, different components of the crystal well are prepared to create and render the virtual scene. More specifically: Light, Camera, Distributor, Material, Builder, Writer and Renderer:

* The **light** places a bottom light (optional) and a top light in director of the plane.
* The **camera** component handles the camera position and angle.
* The **distributor** takes care of positioning the crystals either randomly or by a uniform distribution method.
* The **material** for the crystals is created with a hard-coded value dictionary. The material's index of refraction and brightness are sampled uniformly within a given value range.
* The **builder** "builds" the plane at the bottom of the well and invokes the creation of 3D crystal models within the scene. It also applies the material to the newly created 3D models.
* The **writer** handles the saving of the annotations used for supervised learning.
* The **renderer** sets up Blenders' Cycles to create photorealistic images of the scene.

Finally, the generate_image method of `CrystalWellSimulator` is called. If the "save settings" option was enabled, the `CrystalWellSettings` component writes its dictionary values to a file in the same directory as the output path of the images, which might then be used for headless execution.


### Author and project information
Add-on created by [Sebastian Franz](https://github.com/SebieF) as part of an interdisciplinary project at the 
[chair of biochemical engineering](https://www.epe.ed.tum.de/en/biovt/home/), Technical University of Munich,
in winter term 2021/2022.