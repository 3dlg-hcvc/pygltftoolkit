import json

import numpy as np

from .visual.material import Material, PBRMaterial, TextureMaterial


class Primitive():
    def __init__(self, attributes: dict,
                 vertex_colors: np.ndarray = None,
                 material: Material = None):
        """
        Initialize the Primitive object
        Args:
            attributes: dict, the attributes of the primitive
            vertex_colors: np.ndarray(np.float32), the colors of the vertices (N * 4)
            material: pygltftoolkit.gltfScene.components.visual.Material, the material of the primitive
        Properties:
            attributes: dict, the attributes of the primitive
            vertex_colors: np.ndarray(np.float32), the colors of the vertices (N * 4)
            material: pygltftoolkit.gltfScene.components.visual.Material, the material of the primitive

        """
        self.attributes: dict = attributes
        self.has_normals: bool = False
        if "NORMAL" in self.attributes:
            self.has_normals = True
        self.has_texture: bool = False
        if type(material) is TextureMaterial:
            self.has_texture = True
        self.has_colors: bool = False
        if "COLOR_0" in self.attributes:
            self.has_colors = True
        self.has_baseColorFactor: bool = False
        if not self.has_texture:
            self.has_baseColorFactor = True
        self.vertex_colors: np.ndarray = None
        if vertex_colors is not None:
            if np.any(vertex_colors > 1.0) or np.any(vertex_colors < 0.0):
                raise ValueError("Color must be between 0.0 and 1.0.")
            if vertex_colors.shape[1] != 4:
                raise ValueError("Color must have 4 channels.")
            self.vertex_colors = vertex_colors
            self.has_colors = True
        self.material: Material = material
        if self.has_colors and material is None:
            material = PBRMaterial(baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                                   metallicFactor=1.0,
                                   roughnessFactor=1.0)

    def remove_visuals(self):
        """
        Remove visuals from the primitive
        """
        if self.has_texture:
            self.attributes.pop("TEXCOORD_0")
            self.has_texture = False
        if self.has_colors:
            self.attributes.pop("COLOR_0")
            self.has_colors = False
        if self.has_baseColorFactor:
            self.has_baseColorFactor = False
        self.material = None

    def color_vertices(self, vertex_colors: np.ndarray):
        """
        Color the vertices of the primitive
        Args:
            color: np.ndarray(np.float32), the color of the vertices
        """
        if np.all(vertex_colors > 1.0) or np.all(vertex_colors < 0.0):
            raise ValueError("Color must be between 0.0 and 1.0.")
        self.attributes["COLOR_0"] = vertex_colors
        self.vertex_colors = vertex_colors
        self.has_colors = True

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"Primitive: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"attributes": list(self.attributes.keys()),
                      "has_texture": self.has_texture,
                      "has_colors": self.has_colors,
                      "has_baseColorFactor": self.has_baseColorFactor,
                      "material": self.material.__dict__() if self.material is not None else None}
        return class_dict
