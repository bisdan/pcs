import bpy
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )
import blvcw.crystal_well_global_state as GlobalState


class OBJECT_PT_VIRTUAL_CRYSTAL_WELL(Panel):
    """
    Draws the VCW Panel.
    """
    bl_label = "Virtual Crystal Well"
    bl_idname = "OBJECT_PT_VIRTUAL_CRYSTAL_WELL"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Crystal Well"
    bl_context = "objectmode"

    @classmethod
    def poll(self, context):
        return context.object is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        vcw = scene.virtual_crystal_well

        # (Random) Crystal Props
        # Number crystals
        box = layout.box()
        col = box.column()
        col.label(text="Number of Crystals:")
        col.prop(vcw, "number_crystals")
        col.prop(vcw, "number_crystals_std_dev")
        col.separator()
        # Distributor
        box = layout.box()
        col = box.column()
        col.label(text="Crystal Distribution Method:")
        col.prop(vcw, "crystal_distributor")
        if vcw.crystal_distributor == "DEFAULT":
            # Total crystal area
            row = box.row()
            row.prop(vcw, "total_crystal_area_min")
            row.prop(vcw, "total_crystal_area_max")
            # Crystal area
            row = box.row()
            row.prop(vcw, "crystal_area_min")
            row.prop(vcw, "crystal_area_max")
            # Crystal edges
            row = box.row()
            row.prop(vcw, "crystal_edge_min")
            row.prop(vcw, "crystal_edge_max")
            col = box.column()
            col.prop(vcw, "crystal_aspect_ratio_max")
            col.prop(vcw, "smooth_shading_distributor")
            col.separator()
        elif vcw.crystal_distributor == "RANDOM":
            # Scaling
            col.label(text="Scaling of Crystals:")
            col.prop(vcw, "scaling_crystals_average")
            col.prop(vcw, "scaling_crystals_std_dev")
            col.separator()
            # Rotation
            col.label(text="Rotation of Crystals:")
            col.prop(vcw, "rotation_crystals_average")
            col.prop(vcw, "rotation_crystals_std_dev")
            col.separator()
        # Object Import
        box = layout.box()
        col = box.column()
        if GlobalState.has_rendered:
            col.enabled = False  # Prevent further imports after rendering
        col.label(text="Crystal object:")
        col.prop(vcw, "crystal_object")
        if vcw.crystal_object == "CUSTOM":
            if GlobalState.import_error:
                col.alert = True
                col.label(text="Import path error!")
                col.prop(vcw, "import_path")
            else:
                col.alert = False
                col.prop(vcw, "import_path")
                col = box.column()
                col.prop(vcw, "number_variants")
        col.separator()
        # Crystal Material
        box = layout.box()
        col = box.column()
        col.label(text="Crystal material:")
        col.prop(vcw, "crystal_material")
        if vcw.crystal_material == "GLASS":
            col.label(text="IOR of glass:")
            row = box.row()
            row.prop(vcw, "crystal_material_min_ior")
            row.prop(vcw, "crystal_material_max_ior")
            col = box.column()
            col.label(text="Brightness of glass:")
            row = box.row()
            row.prop(vcw, "crystal_material_min_brightness")
            row.prop(vcw, "crystal_material_max_brightness")
        col.separator()
        # Light
        box = layout.box()
        col = box.column()
        col.label(text="Light:")
        col.prop(vcw, "light_type")
        row = box.row()
        row.prop(vcw, "light_angle_min")
        row.prop(vcw, "light_angle_max")
        row = box.row()
        row.prop(vcw, "use_bottom_light")
        col.separator()
        # Rendering
        box = layout.box()
        col = box.column()
        col.label(text="Rendering:")
        col.prop(vcw, "resolution_x")
        col.prop(vcw, "resolution_y")
        col.prop(vcw, "number_images")
        col.prop(vcw, "update_ui")
        col.prop(vcw, "save_settings")
        col.separator()
        col = box.column()
        if GlobalState.output_directory_error:
            col.alert = True
            col.label(text="Output directory does not exist!")
        elif GlobalState.output_directory_warning:
            col.alert = True
            col.label(text="Output directory is not empty!")
        col.prop(vcw, "output_path")
        col.separator()
        layout.operator("wm.vcw_execute")
        layout.separator()
