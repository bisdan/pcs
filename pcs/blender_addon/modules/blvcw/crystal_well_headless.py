"""
Crystal Well Headless execution.
Should not be called directly but via headless_execution.py.
"""

import os
import sys
from blvcw.crystal_well_components import CrystalWellSettings, CrystalWellLoader
from blvcw.crystal_well_simulation import CrystalWellSimulator


class __CrystalWellHeadlessExecution:
    """
    Performs headless execution with a provided settings file.
    The following steps are performed:
    1. CrystalWellSettings is loaded with the settings file
    2. CrystalWellLoader is generated and imports the crystal object if a custom file is provided
    3. CrystalWellSimulator is called with the classes created before and renders like in the add-on
    """

    def __init__(self, settings_file_path):
        self.settings_file_path = settings_file_path

    def perform_headless_execution(self):
        crystal_well_settings = CrystalWellSettings()
        crystal_well_settings.from_json(self.settings_file_path)

        settings_dict = crystal_well_settings.settings_dict

        crystal_well_loader = CrystalWellLoader(crystal_object=settings_dict["crystal_object"])
        if settings_dict["crystal_object"] == "CUSTOM":
            crystal_well_loader.import_obj(settings_dict["crystal_import_path"], clear_material=True)
            crystal_well_loader.set_number_crystal_variants_per_render(
                number_crystal_variants=settings_dict["number_variants"])
        crystal_well_simulator = CrystalWellSimulator(crystal_well_settings=crystal_well_settings,
                                                      crystal_well_loader=crystal_well_loader)

        counter = 0
        for _ in crystal_well_simulator.generate_image():
            counter += 1
            print("Rendered image number:", counter)

        print("Headless execution finished. Created " + str(counter) + " images.")


argv = sys.argv

if "--settings_file" not in argv:
    print("ERROR: NO SETTINGS FILE PROVIDED")
    exit(1)

settings_file = argv[argv.index("--settings_file") + 1]

if settings_file == "":
    print("ERROR: NO SETTINGS FILE PROVIDED")
    exit(1)
elif not os.path.exists(settings_file):  # Path not found
    print("ERROR: SETTINGS FILE NOT FOUND")
    exit(1)

crystal_well_headless = __CrystalWellHeadlessExecution(settings_file_path=settings_file)
crystal_well_headless.perform_headless_execution()
exit(0)
