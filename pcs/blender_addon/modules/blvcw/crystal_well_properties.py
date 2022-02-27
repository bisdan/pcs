import bpy
import os.path

from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )

import blvcw.crystal_well_global_state as GlobalState
from blvcw.crystal_well_components import CrystalWellLoader


class CrystalWellProperties(PropertyGroup):
    """
    Contains all the properties that are displayed in the UI.
    update_min_max_PROPERTY functions ensure that min_value is always <= max_value
    """
    ### (RANDOM) CRYSTAL VARIABLES ###
    ### Number Crystals ###
    number_crystals: IntProperty(
        name="Max number of crystals",
        description="The average of max numbers of crystals. Is drawn from a normal distribution using "
                    "this value as the average and the standard deviation below. Is always at least 1",
        default=500,
        min=1,
        max=1000
    )
    number_crystals_std_dev: IntProperty(
        name="Std. dev. from # of crystals",
        description="Standard deviation used for the normal distribution calculation of the crystal maximum",
        default=0,
        min=0,
        max=100
    )
    crystal_distributor: EnumProperty(
        name="Crystal Distributor",
        description="Choose method how to distribute crystals",
        items={
            ("DEFAULT", "Default", "Use the default distribution method"),
            ("RANDOM", "Randomized simple", "Use a distribution method based on chance for positioning "
                                            "and normal distributed values for scaling and rotation"),
        },
        default="DEFAULT")
    ### DEFAULT:
    def update_min_max_distributor_default(self, context):
        if self.total_crystal_area_min > self.total_crystal_area_max:
            self.total_crystal_area_min = self.total_crystal_area_max
        if self.total_crystal_area_max < self.total_crystal_area_min:
            self.total_crystal_area_max = self.total_crystal_area_min

        if self.crystal_area_min > self.crystal_area_max:
            self.crystal_area_min = self.crystal_area_max
        if self.crystal_area_max < self.crystal_area_min:
            self.crystal_area_max = self.crystal_area_min

        if self.crystal_edge_min > self.crystal_edge_max:
            self.crystal_edge_min = self.crystal_edge_max
        if self.crystal_edge_max < self.crystal_edge_min:
            self.crystal_edge_max = self.crystal_edge_min

    total_crystal_area_min: FloatProperty(
        name="Min total Crystal Area",
        description="Minimum crystal area covered in image",
        precision=2,
        default=0.05,
        min=0.05,
        max=0.50,
        update=update_min_max_distributor_default
    )
    total_crystal_area_max: FloatProperty(
        name="Max total Crystal Area",
        description="Maximum crystal area covered in image",
        precision=2,
        default=0.50,
        min=0.05,
        max=0.50,
        update=update_min_max_distributor_default
    )
    crystal_area_min: IntProperty(
        name="Min Crystal Area",
        description="Minimum projected crystal area expressed as a fraction of the total image area (**2)",
        default=3,
        min=3,
        max=128,
        update=update_min_max_distributor_default
    )
    crystal_area_max: IntProperty(
        name="Max Crystal Area",
        description="Minimum projected crystal area expressed as a fraction of the total image area (**2)",
        default=128,
        min=3,
        max=128,
        update=update_min_max_distributor_default
    )
    crystal_edge_min: IntProperty(
        name="Min Crystal Edges",
        description="Minimum number of crystal edges",
        default=3,
        min=3,
        max=384,
        update=update_min_max_distributor_default
    )
    crystal_edge_max: IntProperty(
        name="Max Crystal Edges",
        description="Maximum number of crystal edges",
        default=384,
        min=3,
        max=384,
        update=update_min_max_distributor_default
    )
    crystal_aspect_ratio_max: IntProperty(
        name="Aspect ratio max",
        description="Maximum aspect ratio for crystals (/0.7)",
        default=8,
        min=1,
        max=8,
    )
    smooth_shading_distributor: BoolProperty(
        name="Smooth shading",
        description="Use smooth shading in default distributor",
        default=True
    )
    ### RANDOM: Scaling
    scaling_crystals_average: FloatVectorProperty(
        name="Average scale of crystals",
        description="The average of (x, y, z) scaling values used for normal distributed scaling",
        default=(1.0, 1.0, 1.0),
        min=0.05,
        max=5
    )
    scaling_crystals_std_dev: FloatProperty(
        name="Std. dev. from scaling of crystals",
        description="The standard deviation of (x, y, z) used for normal distributed scaling",
        default=0,
        min=0,
        max=0.5
    )
    ### RANDOM: Orientation/Rotation
    rotation_crystals_average: FloatVectorProperty(
        name="Average rotation of crystals",
        description="The average of (x, y, z) rotation values used for normal distributed rotating",
        default=(0.0, 0.0, 0.0),
        min=0,
        max=360.0
    )
    rotation_crystals_std_dev: FloatProperty(
        name="Std. dev. from rotation of crystals",
        description="The standard deviation of (x, y, z) used for normal distributed rotating",
        default=0,
        min=0,
        max=90
    )
    ### IMPORT CRYSTAL OBJECT ###
    def update_variants_max(self, context):
        if GlobalState.crystal_well_loader is not None:
            number_variants_imported = GlobalState.crystal_well_loader.get_number_of_variants()
            if self.number_variants > number_variants_imported:
                self.number_variants = max(1, number_variants_imported)

    def try_import(self, context):
        """
        Tries to import an object file provided in the UI that contains crystals.
        It uses the CrystalWellLoader class and sets it for global availability in global_state.
        If importing fails, import_error is set to True and indicated in the UI.
        """
        if GlobalState.crystal_well_loader is not None:
            GlobalState.crystal_well_loader.clear_import()
            GlobalState.crystal_well_loader = None

        crystal_well_loader = CrystalWellLoader(crystal_object=self.crystal_object)
        if self.crystal_object == "CUSTOM":
            try:
                crystal_well_loader.import_obj(import_path=self.import_path,
                                               clear_material=True)
                GlobalState.import_error = False
                GlobalState.crystal_well_loader = crystal_well_loader
                self.number_variants = crystal_well_loader.get_number_of_variants()
            except Exception as e:
                print(e)
                print("Selected object could not be imported, please provide a correct .obj file path")
                GlobalState.import_error = True
                GlobalState.crystal_well_loader = None

    crystal_object: EnumProperty(
        name="Crystal Object",
        description="Choose crystal object",
        items={
            ("CUSTOM", "Custom object", "Import custom .obj file"),
            ("CUBE", "Generic Cube object", "Use a generic cube object for testing"),
        },
        default="CUSTOM",
        update=try_import)
    import_path: StringProperty(
        name="Import Object",
        description="Choose a valid .obj file that contains one or more crystal meshes",
        default=GlobalState.default_import_path,
        maxlen=1024,
        subtype="FILE_PATH",
        update=try_import
    )
    number_variants: IntProperty(
        name="Number of variants",
        description="Choose between how many different variants from imported file should "
                    "be chosen during vcw creation",
        default=1,
        min=1,
        max=100000,
        update=update_variants_max
    )
    ### MATERIALS ###
    crystal_material: EnumProperty(
        name="Crystal Material",
        description="Choose crystal material",
        items={
            ("GLASS", "Glass material", "Simple glass material (recommended)"),
            ("NONE", "No material", "Does not use any material"),
        },
        default="GLASS")

    def update_min_max_material(self, context):
        if self.crystal_material_min_ior > self.crystal_material_max_ior:
            self.crystal_material_min_ior = self.crystal_material_max_ior
        if self.crystal_material_max_ior < self.crystal_material_min_ior:
            self.crystal_material_max_ior = self.crystal_material_min_ior

        if self.crystal_material_min_brightness > self.crystal_material_max_brightness:
            self.crystal_material_min_brightness = self.crystal_material_max_brightness
        if self.crystal_material_max_brightness < self.crystal_material_min_brightness:
            self.crystal_material_max_brightness = self.crystal_material_min_brightness

    crystal_material_min_ior: FloatProperty(
        name="Min IOR",
        description="Minimum Glass BSDF IOR (index of refraction) - value is drawn for every crystal from min - max",
        precision=3,
        default=1.100,
        min=1.000,
        max=5.000,
        update=update_min_max_material
    )
    crystal_material_max_ior: FloatProperty(
        name="Max IOR",
        description="Maximum Glass BSDF IOR (index of refraction) - value is drawn for every crystal from min - max",
        precision=3,
        default=1.600,
        min=1.000,
        max=5.000,
        update=update_min_max_material
    )
    crystal_material_min_brightness: FloatProperty(
        name="Min Brightness",
        description="Minimum brightness (color value) - value is drawn for every crystal from min - max",
        precision=2,
        default=0.75,
        min=0.75,
        max=0.90,
        update=update_min_max_material
    )
    crystal_material_max_brightness: FloatProperty(
        name="Max Brightness",
        description="Maximum brightness (color value) - value is drawn for every crystal from min - max",
        precision=2,
        default=0.90,
        min=0.75,
        max=0.90,
        update=update_min_max_material
    )
    ### LIGHT ###
    light_type: EnumProperty(
        name="Light Type",
        description="Choose light type for light above plane",
        items={
            ("AREA", "Area light", "Area light (default)"),
            ("SPOT", "Spot light", "Spot light"),
        },
        default="AREA")

    def update_min_max_light(self, context):
        if self.light_angle_min > self.light_angle_max:
            self.light_angle_min = self.light_angle_max
        if self.light_angle_max < self.light_angle_min:
            self.light_angle_max = self.light_angle_min

    light_angle_min: IntProperty(
        name="Minimum light angle",
        description="Light angle for every image gets drawn between [angle_min, angle_max]",
        default=30,
        min=-90,
        max=90,
        update=update_min_max_light
    )
    light_angle_max: IntProperty(
        name="Maximum light angle",
        description="Light angle for every image gets drawn between [angle_min, angle_max]",
        default=60,
        min=-90,
        max=90,
        update=update_min_max_light
    )
    use_bottom_light: BoolProperty(
        name="Use bottom light",
        description="If checked, uses a small AREA bottom light at the level of the crystal well plane (recommended)",
        default=True
    )
    ### RENDERING ###
    resolution_x: IntProperty(
        name="Resolution X",
        description="Width of rendered image",
        default=384,
        min=50,
        max=2000
    )
    resolution_y: IntProperty(
        name="Resolution Y",
        description="Height of rendered image",
        default=384,
        min=50,
        max=2000
    )
    number_images: IntProperty(
        name="Number of images",
        description="Number of images to render (and thus different crystal wells to generate)",
        default=1,
        min=0,
        max=10000
    )
    update_ui: BoolProperty(
        name="Update UI",
        description="Updates UI during rendering (after each vcw was created)",
        default=False
    )
    save_settings: BoolProperty(
        name="Save Settings",
        description="Saves current settings into output path as .json file",
        default=False
    )

    def check_output_path(self, context):
        if os.path.exists(self.output_path):
            GlobalState.output_directory_error = False
            if len(os.listdir(self.output_path)) != 0:
                GlobalState.output_directory_warning = True
            else:
                GlobalState.output_directory_warning = False
        else:
            GlobalState.output_directory_error = True

    output_path: StringProperty(
        name="Output Directory",
        description="Choose a directory where pictures should be saved",
        default=GlobalState.default_output_directory,
        maxlen=1024,
        subtype="DIR_PATH",
        update=check_output_path
    )
