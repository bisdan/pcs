import bpy
import numpy as np
from blvcw.crystal_well_components import CrystalWellLight, CrystalWellCamera, CrystalWellBuilder, \
    CrystalWellRenderer, CrystalWellWriter, CrystalWellSettings
from blvcw.crystal_well_simulation_utils import get_random_euler, get_random_translation, get_normal_distributed_values
from blvcw.crystal_well_distributor import VCWDefaultDistributor, CrystalWellRandomDistributor


class CrystalWellSimulator:
    """
    Main class of the VCW. It connects the front-end UI with the back-end rendering.
    It must be provided a crystal_well_settings and loader object that have been initialized before, either in
    the operator or the headless execution.
    The init methods creates all necessary VCW components.
    After that, the generate_image method can be called to create the images and show the crystal well in blender.
    """

    def __init__(self, crystal_well_settings, crystal_well_loader):

        settings = crystal_well_settings.settings_dict

        self.name = "CrystalWell"
        self.number_of_images = settings["number_images"]
        self.crystal_well_loader = crystal_well_loader

        gamma = 3. / (((settings["camera_distance"] - settings["cw_depth"]) ** 3) - settings["camera_distance"] ** 3)
        plane_length = 2 * (np.tan(settings["field_of_view"] / 2) *
                            (settings["camera_distance"] - settings["cw_depth"]))
        number_crystals = max(1, int(get_normal_distributed_values(mu=settings["number_crystals"],
                                                                   sigma=settings["number_crystals_std_dev"],
                                                                   number_values=1)[0]))

        def random_translation_function():
            return get_random_translation(gamma=gamma,
                                          camera_distance=settings["camera_distance"],
                                          field_of_view=settings["field_of_view"])

        self.camera = CrystalWellCamera(field_of_view=settings["field_of_view"],
                                        camera_distance=settings["camera_distance"],
                                        cw_depth=settings["cw_depth"])
        self.light = CrystalWellLight(light_type=settings["light_type"],
                                      light_angle_min=settings["light_angle_min"],
                                      light_angle_max=settings["light_angle_max"],
                                      plane_length=plane_length,
                                      use_bottom_light=settings["use_bottom_light"],
                                      camera_distance=settings["camera_distance"],
                                      cw_depth=settings["cw_depth"])
        if settings["distributor"] == "RANDOM":
            self.distributor = CrystalWellRandomDistributor(number_crystals=number_crystals,
                                                            scaling_crystals_average=settings[
                                                                "scaling_crystals_average"],
                                                            scaling_crystals_std_dev=settings[
                                                                "scaling_crystals_std_dev"],
                                                            rotation_crystals_average=settings[
                                                                "rotation_crystals_average"],
                                                            rotation_crystals_std_dev=settings[
                                                                "rotation_crystals_std_dev"],
                                                            res_x=settings["res_x"],
                                                            res_y=settings["res_y"],
                                                            random_translation_function=random_translation_function,
                                                            crystal_well_loader=crystal_well_loader)
        elif settings["distributor"] == "DEFAULT":
            self.distributor = VCWDefaultDistributor(max_n_crystals=number_crystals,
                                                     total_crystal_area_min=settings["total_crystal_area_min"],
                                                     total_crystal_area_max=settings["total_crystal_area_max"],
                                                     res_x=settings["res_x"],
                                                     res_y=settings["res_y"],
                                                     crystal_area_min=settings["crystal_area_min"],
                                                     crystal_area_max=settings["crystal_area_max"],
                                                     crystal_edge_min=settings["crystal_edge_min"],
                                                     crystal_edge_max=settings["crystal_edge_max"],
                                                     crystal_aspect_ratio_max=settings["crystal_aspect_ratio_max"],
                                                     smooth_shading=settings["smooth_shading_distributor"],
                                                     random_translation_function=random_translation_function,
                                                     crystal_well_loader=crystal_well_loader,
                                                     cw_depth=settings["cw_depth"])

        self.crystal_well_loader = crystal_well_loader
        self.builder = CrystalWellBuilder(plane_length=plane_length,
                                          cw_depth=settings["cw_depth"],
                                          distributor=self.distributor,
                                          material_name=settings["crystal_material_name"],
                                          material_min_ior=settings["crystal_material_min_ior"],
                                          material_max_ior=settings["crystal_material_max_ior"],
                                          material_min_brightness=settings["crystal_material_min_brightness"],
                                          material_max_brightness=settings["crystal_material_max_brightness"])

        self.renderer = CrystalWellRenderer(number_frames=settings["n_frames"],
                                            number_threads=settings["number_threads"],
                                            res_x=settings["res_x"],
                                            res_y=settings["res_y"],
                                            output_path=settings["output_path"])

        self.json_writer = CrystalWellWriter(output_path=settings["output_path"])

        self.collection_cw = None

    def _clear_scene(self):
        for collection in bpy.data.collections:
            if collection.name != self.crystal_well_loader.collection_name:
                bpy.data.collections.remove(collection)

        # delete vcw objects
        delete_names = [obj.name for obj in bpy.data.objects if obj.name.startswith("vcw_")]
        for del_name in delete_names:
            bpy.data.objects.remove(bpy.data.objects[del_name], do_unlink=True)

        # delete meshes
        delete_names = [me.name for me in bpy.data.meshes if me.name.startswith("vcw_")]
        for del_name in delete_names:
            bpy.data.meshes.remove(bpy.data.meshes[del_name], do_unlink=True)

        # delete materials
        delete_names = [mat.name for mat in bpy.data.materials if "CrystalMaterial" in mat.name]
        for del_name in delete_names:
            bpy.data.materials.remove(bpy.data.materials[del_name], do_unlink=True)

    def generate_image(self):
        """
        Main generator method that renders each image and creates the according segmentation file.
        On success, it yields true so that the UI can be updated.
        If number_of_images is 0, only the VCW in blender is updated, images are not rendered or saved.
        """
        render = True
        if self.number_of_images == 0:
            render = False
            self.number_of_images = 1  # To start for loop
        for i in range(1, self.number_of_images + 1):
            self._clear_scene()
            self._setup_crystal_well()
            self.crystal_well_loader.setup()
            self.camera.setup()
            self.light.setup()
            self.renderer.setup()
            segmentations = self.builder.setup()
            yield True  # Update UI
            if render:
                image_name = self.renderer.render_image(image_index=i)
                self.json_writer.write_json(image_name=image_name, polygons=segmentations)

    def _setup_crystal_well(self):
        """
        Creates a new collection for the VCW.
        """
        bpy.ops.object.collection_instance_add("INVOKE_DEFAULT")

        collection = bpy.data.collections.new(self.name)
        bpy.context.scene.collection.children.link(collection)
        self.collection_cw = bpy.context.view_layer.layer_collection.children[self.name]
        bpy.context.view_layer.active_layer_collection = self.collection_cw
