import json

import numpy as np

from .image import TextureImage


class Material():
    """
    Abstract class for Material objects
    """
    def __init__(self) -> None:
        pass


class TextureMaterial(Material):
    def __init__(self, uv: np.ndarray, image: TextureImage, sampler: int = None) -> None:
        """
        Initialize the TextureMaterial object
        Args:
            texture: str, the path to the texture
        Properties:
            texture: str, the path to the texture
        """
        self.texture: str = image
        self.uv: np.ndarray = uv
        self.sampler: int = sampler

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"TextureMaterial: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"texture": self.texture.__dict__()}
        return class_dict


class PBRMaterial(Material):
    def __init__(self, baseColorFactor: np.ndarray, metallicFactor: float, roughnessFactor: float) -> None:
        """
        Initialize the PBRMaterial object
        Args:
            baseColorFactor: np.ndarray(np.float32), the base color factor
            metallicFactor: float, the metallic factor
            roughnessFactor: float, the roughness factor
        Properties:
            baseColorFactor: np.ndarray(np.float32), the base color factor
            metallicFactor: float, the metallic factor
            roughnessFactor: float, the roughness factor
        """
        if type(baseColorFactor) is not np.ndarray:
            baseColorFactor = np.asarray(baseColorFactor, dtype=np.float32)
        self.baseColorFactor: np.ndarray = baseColorFactor
        self.metallicFactor: float = metallicFactor
        self.roughnessFactor: float = roughnessFactor

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"PBRMaterial: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"baseColorFactor": self.baseColorFactor.tolist(),
                      "metallicFactor": self.metallicFactor,
                      "roughnessFactor": self.roughnessFactor}
        return class_dict
