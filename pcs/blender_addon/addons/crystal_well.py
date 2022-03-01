bl_info = {
    "name": "Crystal Well",
    "description": "Creating a virtual crystal well.",
    "author": "SebieF",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D View > CrystalWell",
    "warning": "",
    "wiki_url": "https://github.com/bisdan/pcs",
    "tracker_url": "",
    "category": "Development"
}

import bpy
from bpy.props import PointerProperty
from bpy.utils import register_class, unregister_class

from blvcw.crystal_well_properties import CrystalWellProperties
from blvcw.crystal_well_ui import OBJECT_PT_VIRTUAL_CRYSTAL_WELL
from blvcw.crystal_well_operator import WM_OT_VIRTUAL_CRYSTAL_WELL

classes = (
    CrystalWellProperties,
    OBJECT_PT_VIRTUAL_CRYSTAL_WELL,
    WM_OT_VIRTUAL_CRYSTAL_WELL,
)


def register():
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.virtual_crystal_well = PointerProperty(type=CrystalWellProperties)


def unregister():
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.virtual_crystal_well


if __name__ == "__main__":
    register()
