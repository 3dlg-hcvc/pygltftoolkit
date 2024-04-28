import json

import numpy as np

from .visual.material import Material


class Primitive():
    def __init__(self, attributes: dict,
                 vertex_colors: np.ndarray = None,
                 material: Material = None):
        """
        Initialize the Primitive object
        Args:
            attributes: dict, the attributes of the primitive
            material: pygltftoolkit.gltfScene.components.visual.Material, the material of the primitive
        Properties:
            attributes: dict, the attributes of the primitive
            material: pygltftoolkit.gltfScene.components.visual.Material, the material of the primitive
        """
        self.attributes: dict = attributes
        self.has_normals: bool = False
        if "NORMAL" in self.attributes:
            self.has_normals = True
        self.has_texture: bool = False
        if "TEXCOORD_0" in self.attributes:
            self.has_texture = True
        self.has_colors: bool = False
        if "COLOR_0" in self.attributes:
            self.has_colors = True
        self.has_baseColorFactor: bool = False
        if not self.has_texture and not self.has_colors:
            self.has_baseColorFactor = True
        if vertex_colors is not None:
            self.has_colors = True
        self.material: Material = material

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"Primitive: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"attributes": list(self.attributes.keys()),
                      "has_texture": self.has_texture,
                      "has_colors": self.has_colors,
                      "has_baseColorFactor": self.has_baseColorFactor,
                      "material": self.material.__dict__()}
        return class_dict
