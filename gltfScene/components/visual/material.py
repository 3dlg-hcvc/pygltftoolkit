import numpy as np

from .image import TextureImage


class Material():
    """
    Abstract class for Material objects
    """
    def __init__(self) -> None:
        pass


class TextureMaterial(Material):
    def __init__(self, uv: np.ndarray, image: TextureImage) -> None:
        """
        Initialize the TextureMaterial object
        Args:
            texture: str, the path to the texture
        Properties:
            texture: str, the path to the texture
        """
        self.texture: str = image
        self.uv: np.ndarray = uv


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
        self.baseColorFactor: np.ndarray = baseColorFactor
        self.metallicFactor: float = metallicFactor
        self.roughnessFactor: float = roughnessFactor
