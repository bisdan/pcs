# PCS Blender Add-On

## Installation

Please refer to the main [README](https://github.com/bisdan/pcs/blob/main/README.md#pcs-blender-add-on) 
for installation instructions.

## How-To

After the installation, the main panel of the add-on can be found in object mode (default mode after start-up) by
pressing "N" or opening the 3D-viewport sidebar. Here you should find an entry named "Crystal Well".
In this panel, you can adjust every property used for the creation of the crystal well. Please refer to the 
more detailed description of each property by hovering over it.
Note that if you are using a default .obj file for the crystal meshes, you will see an import error until a valid
file has been loaded.
Same applies to the output directory: There will be an error as long as the provided output path is not found.
If it is found, yet it is not empty, a warning will be displayed. 
**Images that have the same name in this directory will be overwritten by the new images.**

### Hints & Tipps
* The *Update UI* option lets you take a look at the resulting VCW after every render. This can be especially 
useful if you want to render multiple images and want to take a look at the intermediate crystal wells in blender.
* If you want to use a *default import or output directory*, you can change the default values in 
modules/blvcw/crystal_well_global_state.py.

### Headless execution

Please refer to the main [README](https://github.com/bisdan/pcs/blob/main/README.md#pcs-blender-add-on) 
for headless execution instructions.

*Note: The default settings require up to a few minutes for a single image. 
The PCS data sets were created by running multiple worker processes on many machines.*

*Note: It is not possible to load existing settings directly into the blender addon, it is only possible
to save them from the values provided in the UI. The loading task is handled by the headless execution.*

## Troubleshooting
* Using non-default values may cause the program to not function as intended. 
Especially odd resolution values (e.g. res_x >> res_y) can cause problems with the default distributor.
If you encounter such a situation, feel free to let us know and create an issue!

## Architecture of the addon
This section will briefly describe the design and architecture of the blender add-on. Of course, implementation and
design have been limited by the restrictions that blender puts on its add-ons. 
Overall, it was tried to use an architecture as modular as possible. That is why for every distinct task, a different
component was introduced that should handle only its specific task.

The UI (found in properties and ui) manage the setting of the properties. When the user hits the "Render" button, the
operator comes into play and all these settings are saved in the `CrystalWellSettings` component. 
Furthermore, the `CrystalWellLoader` component is created to manage import and loading of single crystals.
The operator task can also be replaced by the `CrystalWellHeadlessExecution` class, which enables headless execution.
It should, however, only be called via the headless_execution.py script. 
Both the operator and the headless execution class would then continue to create a CrystalWellSimulator component,
which orchestrates the different components of the crystal well and puts them together. More specific, it creates Light,
Camera, Distributor, Material, Builder, Writer and Renderer:

* The **light** places a bottom light (optional) and a top light in director of the plane.
* The **camera** component handles the camera position and angle.
* The **distributor** takes care of positioning the crystals either randomly or by a uniform distribution method.
* The **material** for the crystals is created with a hard-coded value dictionary. The material's IOR and brightness
can be shuffled, however.
* The **builder** "builds" the plane at the bottom of the well and invokes the creation of crystals. It also applies
the material to the newly created crystals.
* The **writer** handles the saving of the segmentation of the crystal polygons.
* Finally, the **renderer** cares about actually rendering and saving the image.

After creation of these elements, the generate_image method of `CrystalWellSimulator` is called. It is a generator 
method which enables the UI to be updated after each rendered image. Finally, if the user selected the "save settings"
check box in the UI, the `CrystalWellSettings` component writes its dictionary values to a file in the same directory 
as the output path of the images.
The only scripts that have not been mentioned yet are `crystal_well_global_state` and `crystal_well_simulation_utils`. 
These are not components in itself. The global_state file saves default paths and global state variables to handle
the UI in a dynamic way. The simulation_utils file solely provides utility functions that are used by other components.

### Author and project information
Add-On created by [Sebastian Franz](https://github.com/SebieF) as part of an interdisciplinary project at the 
[chair of biochemical engineering](https://www.epe.ed.tum.de/en/biovt/home/), Technical University Munich,
in winter term 2021/2022.