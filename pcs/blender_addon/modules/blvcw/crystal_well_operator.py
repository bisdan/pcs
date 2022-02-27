import bpy
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )
import blvcw.crystal_well_global_state as GlobalState
from blvcw.crystal_well_components import CrystalWellLoader, CrystalWellSettings
from blvcw.crystal_well_simulation import CrystalWellSimulator


class WM_OT_VIRTUAL_CRYSTAL_WELL(Operator):
    """
    VCW operator that is called when pressing the "Render" button in the UI panel.
    It makes sure that a crystal_well_loader was created and that the crystal_well_settings dict is filled
    before the main simulation object is created.
    Will not execute if either an import error exists or no output_directory was chosen for rendering output.
    """
    bl_label = "Render"
    bl_idname = "wm.vcw_execute"

    def execute(self, context):
        scene = context.scene
        vcw = scene.virtual_crystal_well

        crystal_well_loader = GlobalState.crystal_well_loader
        if crystal_well_loader is None:
            if vcw.crystal_object == "CUSTOM":
                # Import error
                return {"CANCELLED"}
            else:
                crystal_well_loader = CrystalWellLoader(crystal_object=vcw.crystal_object)
        crystal_well_loader.set_number_crystal_variants_per_render(number_crystal_variants=vcw.number_variants)
        if GlobalState.output_directory_error and vcw.number_images != 0:
            print("No existing output directory chosen!")
            return {"CANCELLED"}

        if GlobalState.output_directory_warning:
            print("WARNING: Saving to non-empty directory")

        scaling_crystals_average = (vcw.scaling_crystals_average[0],  # Necessary because bpy object is not serializable
                                    vcw.scaling_crystals_average[1],
                                    vcw.scaling_crystals_average[2])
        rotation_crystals_average = (vcw.rotation_crystals_average[0],
                                     vcw.rotation_crystals_average[1],
                                     vcw.rotation_crystals_average[2])

        crystal_well_settings = CrystalWellSettings(number_crystals=vcw.number_crystals,
                                                    number_crystals_std_dev=vcw.number_crystals_std_dev,
                                                    crystal_object=vcw.crystal_object,
                                                    distributor=vcw.crystal_distributor,
                                                    total_crystal_area_min=vcw.total_crystal_area_min,
                                                    total_crystal_area_max=vcw.total_crystal_area_max,
                                                    crystal_area_min=vcw.crystal_area_min**2,
                                                    crystal_area_max=vcw.crystal_area_max**2,
                                                    crystal_edge_min=vcw.crystal_edge_min,
                                                    crystal_edge_max=vcw.crystal_edge_max,
                                                    crystal_aspect_ratio_max=vcw.crystal_aspect_ratio_max/0.7,
                                                    smooth_shading_distributor=vcw.smooth_shading_distributor,
                                                    scaling_crystals_average=scaling_crystals_average,
                                                    scaling_crystals_std_dev=vcw.scaling_crystals_std_dev,
                                                    rotation_crystals_average=rotation_crystals_average,
                                                    rotation_crystals_std_dev=vcw.rotation_crystals_std_dev,
                                                    crystal_material_name=vcw.crystal_material,
                                                    crystal_material_min_ior=vcw.crystal_material_min_ior,
                                                    crystal_material_max_ior=vcw.crystal_material_max_ior,
                                                    crystal_material_min_brightness=vcw.crystal_material_min_brightness,
                                                    crystal_material_max_brightness=vcw.crystal_material_max_brightness,
                                                    light_type=vcw.light_type,
                                                    light_angle_min=vcw.light_angle_min,
                                                    light_angle_max=vcw.light_angle_max,
                                                    use_bottom_light=vcw.use_bottom_light,
                                                    res_x=vcw.resolution_x,
                                                    res_y=vcw.resolution_y,
                                                    crystal_import_path=vcw.import_path,
                                                    output_path=vcw.output_path,
                                                    number_variants=vcw.number_variants,
                                                    number_images=vcw.number_images)

        if vcw.save_settings and not GlobalState.output_directory_error:
            crystal_well_settings.write_json()

        crystal_well_settings.print_settings()
        GlobalState.has_rendered = True
        crystal_well_simulator = CrystalWellSimulator(crystal_well_settings=crystal_well_settings,
                                                      crystal_well_loader=crystal_well_loader)

        for ui_update in crystal_well_simulator.generate_image():
            if vcw.update_ui and ui_update:
                # Update VCW in blender after each image is rendered
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        return {"FINISHED"}
