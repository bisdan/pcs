import bpy
import random
from abc import ABC, abstractmethod


class _CrystalMaterial(ABC):
    """
    Abstract class to handle different classes of material (Crystal / Plane).
    """
    @abstractmethod
    def __init__(self, material_name):
        self.material_name = material_name
        self.base_name = material_name
        self.properties, self.color_ramp, self.links = MaterialsContainer.get_material(name=material_name)
        self.shuffle = MaterialsContainer.get_shuffle(name=material_name)
        self.material = None
        self.number = 0
        self._create()

    def apply(self, blender_object):
        blender_object.data.materials.append(self.material)

    def _duplicate_with_new_number(self):
        """
        Used when one a-n specific value(s) of the material is or are shuffled
        """
        self.number += 1
        self.material_name = self.base_name + str(self.number)
        self._create()

    def _create(self):
        """
        Creates a new material by iterating over its value dictionaries (properties, links, color ramp)
        and setting the according values
        """
        if self.material_name not in bpy.data.materials:
            self.material = bpy.data.materials.new(self.material_name)
            self.material.use_nodes = True

            if "Principled BSDF" not in self.properties.keys():
                principled_bsdf_to_delete = self.material.node_tree.nodes["Principled BSDF"]
                self.material.node_tree.nodes.remove(principled_bsdf_to_delete)

            for node_name, node_properties in self.properties.items():
                if node_name in self.material.node_tree.nodes:
                    node = self.material.node_tree.nodes.get(node_name)
                else:
                    node = self.material.node_tree.nodes.new(node_name)

                for input_key, value in node_properties.items():
                    if input_key == "distribution":
                        node.distribution = value
                    else:
                        node.inputs[int(input_key)].default_value = value

            for color_ramp, node_properties in self.color_ramp.items():
                if color_ramp in self.material.node_tree.nodes:
                    node = self.material.node_tree.nodes.get(color_ramp)
                else:
                    node = self.material.node_tree.nodes.new(color_ramp)
                for element_key, value in node_properties.items():
                    setattr(node.color_ramp.elements[0], element_key, value)

            for node_name_left, link_properties in self.links.items():
                if link_properties["type"] == "overwrite":
                    node_left = self.material.node_tree.nodes.get(node_name_left)
                    for link in node_left.outputs[0].links:
                        self.material.node_tree.links.remove(link)

                link = self.material.node_tree.links.new

                node_left = self.material.node_tree.nodes.get(node_name_left)
                node_right = self.material.node_tree.nodes.get(link_properties["link_target"])
                link(node_left.outputs[link_properties["link_start_output"]],
                     node_right.inputs[link_properties["link_target_input"]])

        else:
            self.material = bpy.data.materials[self.material_name]


class CrystalMaterialGlass(_CrystalMaterial):
    def __init__(self, material_name="GLASS", min_ior=1.1, max_ior=1.6, min_brightness=0.75, max_brightness=0.9):
        material_name = "CrystalMaterial" + material_name
        self.min_ior = min_ior
        self.max_ior = max_ior
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        super().__init__(material_name=material_name)

    def shuffle_ior_and_brightness(self):
        """
        Draws a new value for ior between self.min_ior and self.max_ior.
        Also draws a new value for brightness between self.min_brightness and self.max_brightness.
        Brightness is set by manipulating the color value of the Glass BSDF Node.
        """
        if self.shuffle:
            ior = random.uniform(self.min_ior, self.max_ior)
            brightness = random.uniform(self.min_brightness, self.max_brightness)
            if "ShaderNodeBsdfGlass" in self.properties.keys():
                self.properties["ShaderNodeBsdfGlass"]["0"] = (brightness, brightness, brightness, 1)
                self.properties["ShaderNodeBsdfGlass"]["1"] = 0  # roughness
                self.properties["ShaderNodeBsdfGlass"]["2"] = ior
                self._duplicate_with_new_number()

    """
    def shuffle_roughness(self):
        roughness = random.uniform(0.15, 0.65)
        if "ShaderNodeBsdfGlass" in self.properties.keys():
            self.properties["ShaderNodeBsdfGlass"]["1"] = roughness
            self._duplicate_with_new_number()
        elif "Principled BSDF" in self.properties.keys():
            self.properties["Principled BSDF"]["7"] = roughness
            self._duplicate_with_new_number()
    """


class PlaneMaterial(_CrystalMaterial):
    def __init__(self, material_name="PlaneMaterial"):
        super().__init__(material_name=material_name)

    """
    def shuffle_transparency(self):
        old_color = self.properties["Principled BSDF"]["0"]
        new_color = (old_color[0], old_color[1], old_color[2], random.uniform(0.8, 1.0))
        self.properties["Principled BSDF"]["0"] = new_color
        self._duplicate_with_new_number()
    """


class MaterialsContainer:
    """
    Contains properties for default build-in materials.
    Can be extended by adding a new material to the get_material method.
    Note that node names (e.g. ShaderNodeBsdfGlass) often differ in blender from their name that is used e.g. for links
    (like "Glass BSDF").
    """
    @staticmethod
    def get_shuffle(name):
        if name in ["CrystalMaterialGLASS"]:
            return True
        else:
            return False

    @staticmethod
    # IOR can only be set for CrystalMaterialGLASS
    def get_material(name):
        properties, color_ramp, links = {}, {}, {}

        if name == "CrystalMaterialGLASS":
            properties = {
                "ShaderNodeBsdfGlass": {
                    "distribution": "BECKMANN",
                    "0": (1.0, 1.0, 1.0, 1.0),  # Color
                    "1": 0.3,  # Roughness
                    "2": 1.1  # IOR (default = 1.050)
                },
            }
            color_ramp = {}
            links = {
                "Glass BSDF": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Material Output",
                    "link_target_input": 0
                },
            }
        elif name == "PlaneMaterial":
            properties = {
                "Principled BSDF": {
                    "0": (0.35, 0.35, 0.35, 0.8)  # Color
                },
            }
            color_ramp = {}
            links = {}
        elif name == "CrystalMaterialOLD":  # Deprecated, not included in add-on UI
            properties = {
                "Principled BSDF": {
                    "5": 0.5,  # Specular
                    "7": 0.05,  # Roughness
                    "11": 0.5,  # Sheen Tint
                    "13": 0.03,  # Clearcoat Roughness
                    "14": 1.45,  # IOR
                    "15": 0.0,  # Transmission (changeable)
                    "16": 0.0,  # Transmission Roughness (changeable)
                },
                "ShaderNodeBsdfTransparent": {
                    "0": (1, 1, 1, 1)  # Color
                },
                "ShaderNodeMixShader": {
                    "0": 0.627,  # Fac
                }
            }
            color_ramp = {}
            links = {
                "Principled BSDF": {
                    "type": "overwrite",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 1
                },
                "Transparent BSDF": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 2
                },
                "Mix Shader": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Material Output",
                    "link_target_input": 0
                },
            }
        elif name == "CrystalMaterialColorRamp":  # Deprecated, not included in add-on UI
            properties = {
                "ShaderNodeBsdfGlass": {
                    "distribution": "BECKMANN",
                    "0": (1.0, 1.0, 1.0, 1.0),  # Color
                    "1": 0.3,  # Roughness
                    "2": 1.005
                },
                "ShaderNodeBsdfGlossy": {
                    "distribution": "BECKMANN",
                    "0": (1.0, 1.0, 1.0, 1.0),  # Color
                    "1": 0.05,  # Roughness
                },
                "ShaderNodeLayerWeight": {
                    "0": 0.3,  # blend
                },
                "ShaderNodeMixShader": {

                },
                "ShaderNodeObjectInfo": {

                },
                "ShaderNodeVolumeAbsorption": {
                    "1": 0.0  # density
                }
            }
            color_ramp = {
                "ShaderNodeValToRGB": {  # elements
                    "color": (1.0, 0.0, 0.0, 1.0),
                    "position": 0.041
                },
            }
            links = {
                "Layer Weight": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 0
                },
                "Glass BSDF": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 1
                },
                "Glossy BSDF": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 2
                },
                "Mix Shader": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Material Output",
                    "link_target_input": 0
                },
                "Object Info": {
                    "type": "new",
                    "link_start_output": 4,
                    "link_target": "ColorRamp",
                    "link_target_input": 0
                },
                "ColorRamp": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Volume Absorption",
                    "link_target_input": 0
                },
                "Volume Absorption": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Material Output",
                    "link_target_input": 1
                }
            }
        elif name == "CrystalMaterialGLASSCOMPLEX":  # Deprecated, not included in add-on UI
            properties = {
                "ShaderNodeBsdfGlass": {
                    "distribution": "BECKMANN",
                    "0": (1.0, 1.0, 1.0, 1.0),  # Color
                    "1": 0.3,  # Roughness
                    "2": 1.005
                },
                "ShaderNodeBsdfGlossy": {
                    "distribution": "BECKMANN",
                    "0": (1.0, 1.0, 1.0, 1.0),  # Color
                    "1": 0.05,  # Roughness
                },
                "ShaderNodeLayerWeight": {
                    "0": 0.3,  # blend
                },
                "ShaderNodeMixShader": {

                },
            }
            color_ramp = {}
            links = {
                "Layer Weight": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 0
                },
                "Glass BSDF": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 1
                },
                "Glossy BSDF": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Mix Shader",
                    "link_target_input": 2
                },
                "Mix Shader": {
                    "type": "new",
                    "link_start_output": 0,
                    "link_target": "Material Output",
                    "link_target_input": 0
                },
            }

        return properties, color_ramp, links
