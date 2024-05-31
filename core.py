import os
import tempfile

import pygltflib

from .gltfScene import gltfScene


def load(
    path: str,
    stk_segmentation: str = None,
    stk_articulation: str = None,
    stk_precomputed_segmentation: str = None
) -> gltfScene:
    """
    Load the glTF 2.0 file. Allows to load the segmentation and articulation annotations as produced by the STK.
    Args:
        path: string, the path to the glTF 2.0 file
        stk_segmentation: string, the path to the segmentation annotations produced by the STK. Defaults to None.
        stk_articulation: string, the path to the articulation annotations produced by the STK. Defaults to None.
    Returns:
        scene: pygltftoolkit.gltfScene object, the glTF 2.0 scene.
    """
    scene = pygltflib.GLTF2().load(path)

    # We support only a single scene in glTF file.
    # Multiple scenes are rarely used and it was even proposed to remove them from the glTF 2.0 specification.
    # See https://github.com/KhronosGroup/glTF/issues/1542
    if len(scene.scenes) > 1:
        raise ValueError("Only one scene in the glTF file is supported.")

    # Please use .glb, we will handle .gltf with an ugly trick
    if path.endswith(".gltf"):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            scene.save_binary(temp_file.name)
            temp_file_path = temp_file.name
        scene = pygltflib.GLTF2().load(temp_file_path)
        os.remove(temp_file_path)

    gltf = gltfScene(scene)

    # Load the segmentation and articulation annotations
    if stk_segmentation is not None:
        # No support yet
        gltf.load_stk_segmentation(stk_segmentation)
    if stk_articulation is not None:
        if stk_segmentation is None:
            raise ValueError("Please provide the segmentation annotations as well.")
        # No support yet
        gltf.load_stk_articulation(stk_articulation)
    if stk_precomputed_segmentation is not None:
        # No support yet
        gltf.load_stk_precomputed_segmentation(stk_precomputed_segmentation)

    return gltf
