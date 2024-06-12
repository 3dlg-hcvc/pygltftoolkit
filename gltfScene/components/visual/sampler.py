import numpy as np
from PIL import Image


class Sampler():
    def __init__(self, magFilter=None, minFilter=None, wrapS=10497, wrapT=10497, name=None, extensions=None, extras=None) -> None:
        """
        Initialize the Sampler object
        Args:
            magFilter: int, the magnification filter
            minFilter: int, the minification filter
            wrapS: int, the wrapS mode
            wrapT: int, the wrapT mode
            name: str, the name of the sampler
            extensions: dict, the extensions
            extras: dict, the extras
        Properties:
            magFilter: int, the magnification filter
            minFilter: int, the minification filter
            wrapS: int, the wrapS mode
            wrapT: int, the wrapT mode
            name: str, the name of the sampler
            extensions: dict, the extensions
            extras: dict, the extras
        """
        three_wrap_lookup = {10497: 1000, 33071: 1001, 33648: 1002}
        self.magFilter = magFilter
        self.minFilter = minFilter
        self.wrapS = wrapS
        self.three_wrapS = three_wrap_lookup[wrapS]
        self.wrapT = wrapT
        self.three_wrapT = three_wrap_lookup[wrapT]
        self.name = name
        self.extensions = extensions
        self.extras = extras

    def sample_from_barycentric(self, faces: np.ndarray, barycentric_coordinates: np.ndarray, uv: np.ndarray, img: Image) -> np.ndarray:
        """
        Sample from the barycentric coordinates
        Args:
            faces: np.ndarray(np.int32), the faces of the primitive
            barycentric_coordinates: np.ndarray(np.float32), the barycentric coordinates
            uv: np.ndarray(np.float32), the uv coordinates
            img: PIL.Image, the image
        Returns:
            np.ndarray(np.float32), the sampled values
        """
        # Interpolate U and V from barycentric coordinates
        u = np.sum(barycentric_coordinates * uv[faces, 0], axis=1)
        v = np.sum(barycentric_coordinates * uv[faces, 1], axis=1)
        # Process out of bounds according to wrap mode
        if self.wrapS == 10497:
            u[u > 1] = u[u > 1] % 1
            u[u < 0] = u[u < 0] % 1
        elif self.wrapS == 33071:
            u[u > 1] = 1
            u[u < 0] = 0
        elif self.wrapS == 33648:
            u[u > 1] = 1 - u[u > 1] % 1
            u[u < 0] = -u[u < 0] % 1

        if self.wrapT == 10497:
            v[v > 1] = v[v > 1] % 1
            v[v < 0] = v[v < 0] % 1
        elif self.wrapT == 33071:
            v[v > 1] = 1
            v[v < 0] = 0
        elif self.wrapT == 33648:
            v[v > 1] = 1 - v[v > 1] % 1
            v[v < 0] = -v[v < 0] % 1

        # Subtract epsilon to avoid out of bounds
        u -= np.finfo(np.float32).eps
        v -= np.finfo(np.float32).eps

        # Sample the image
        u = (u * img.width).astype(np.int32)
        v = (v * img.height).astype(np.int32)

        return np.array(img, dtype=np.float32)[v, u] / 255.0
