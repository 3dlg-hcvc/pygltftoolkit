import os
import json
import struct
import tempfile
import time

from pygltflib import GLTF2

from .gltfScene import gltfScene


_SPARSE_PART_PLACEHOLDER = {"__pygltftoolkit_sparse_parts_connectivity_null__": True}
_GLTF_NUMERIC_ARRAY_KEYS = {
    "baseColorFactor",
    "diffuseFactor",
    "emissiveFactor",
    "matrix",
    "max",
    "min",
    "rotation",
    "scale",
    "specularFactor",
    "translation",
    "weights",
}


def _replace_sparse_parts_connectivity_nulls(payload):
    """Replace null sparse part slots with a decoder-safe placeholder.

    STK stores ``scene.extras.partsConnectivity.parts`` as a pid-indexed sparse
    array.  Since valid part ids generally start at 1, slot 0 can be ``null``.
    pygltflib's dataclasses-json decoder does not tolerate that null inside the
    arbitrary extras payload, so we temporarily replace it while decoding and
    restore it on the loaded GLTF2 object.
    """
    changed = False
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        return changed
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        extras = scene.get("extras")
        if not isinstance(extras, dict):
            continue
        parts_connectivity = extras.get("partsConnectivity")
        if not isinstance(parts_connectivity, dict):
            continue
        parts = parts_connectivity.get("parts")
        if not isinstance(parts, list):
            continue
        for index, part in enumerate(parts):
            if part is None:
                parts[index] = dict(_SPARSE_PART_PLACEHOLDER)
                changed = True
    return changed


def _restore_sparse_parts_connectivity_nulls(scene):
    for gltf_scene in scene.scenes or []:
        extras = gltf_scene.extras
        if not isinstance(extras, dict):
            continue
        parts_connectivity = extras.get("partsConnectivity")
        if not isinstance(parts_connectivity, dict):
            continue
        parts = parts_connectivity.get("parts")
        if not isinstance(parts, list):
            continue
        for index, part in enumerate(parts):
            if isinstance(part, dict) and part.get("__pygltftoolkit_sparse_parts_connectivity_null__"):
                parts[index] = None


def _replace_decoder_unsafe_numeric_nulls(value, parent_key=None):
    """Replace invalid null entries in typed glTF numeric arrays.

    Some STK exports contain JSON ``null`` in arrays that pygltflib decodes as
    ``List[float]``.  That is invalid glTF, but treating the missing numeric as
    zero lets the rest of the generated asset load and keeps the exporter from
    aborting the whole batch.
    """
    changed = False
    if isinstance(value, dict):
        for key, child in value.items():
            child_changed = _replace_decoder_unsafe_numeric_nulls(child, key)
            changed = changed or child_changed
        return changed
    if isinstance(value, list):
        if parent_key in _GLTF_NUMERIC_ARRAY_KEYS:
            for index, child in enumerate(value):
                if child is None:
                    value[index] = 0.0
                    changed = True
                else:
                    child_changed = _replace_decoder_unsafe_numeric_nulls(child, parent_key)
                    changed = changed or child_changed
        else:
            for child in value:
                child_changed = _replace_decoder_unsafe_numeric_nulls(child, parent_key)
                changed = changed or child_changed
    return changed


def _load_with_sparse_parts_connectivity(path):
    try:
        return GLTF2().load(path)
    except TypeError as exc:
        if "NoneType" not in str(exc):
            raise

    if path.endswith(".glb"):
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 20:
            raise
        magic, version, _ = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67 or version != 2:
            raise
        offset = 12
        chunks = []
        changed = False
        while offset + 8 <= len(data):
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset: offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                text = chunk.decode("utf-8").rstrip("\x00 \t\r\n")
                payload = json.loads(text)
                changed = _replace_sparse_parts_connectivity_nulls(payload)
                changed = _replace_decoder_unsafe_numeric_nulls(payload) or changed
                if changed:
                    chunk = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    chunk += b" " * ((4 - (len(chunk) % 4)) % 4)
            chunks.append((chunk_type, chunk))
        if not changed:
            raise
        total_length = 12 + sum(8 + len(chunk) for _, chunk in chunks)
        rebuilt = bytearray(struct.pack("<III", magic, version, total_length))
        for chunk_type, chunk in chunks:
            rebuilt.extend(struct.pack("<II", len(chunk), chunk_type))
            rebuilt.extend(chunk)
        scene = GLTF2().load_from_bytes(bytes(rebuilt))
        _restore_sparse_parts_connectivity_nulls(scene)
        return scene

    if path.endswith(".gltf"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        changed = _replace_sparse_parts_connectivity_nulls(payload)
        changed = _replace_decoder_unsafe_numeric_nulls(payload) or changed
        if not changed:
            raise
        scene = GLTF2().from_json(json.dumps(payload))
        _restore_sparse_parts_connectivity_nulls(scene)
        return scene

    raise


def load(
    path: str,
    stk_segmentation: str = None,
    stk_articulation: str = None,
    stk_precomputed_segmentation: str = None,
    annotated: bool = False
) -> gltfScene:
    """
    Load the glTF 2.0 file. Allows to load the segmentation and articulation annotations as produced by the STK.
    Args:
        path: string, the path to the glTF 2.0 file
        stk_segmentation: string, the path to the segmentation annotations produced by the STK. Defaults to None.
        stk_articulation: string, the path to the articulation annotations produced by the STK. Defaults to None.
        annotated: bool, whether the object has STK annotations embedded. Defaults to False.
    Returns:
        scene: pygltftoolkit.gltfScene object, the glTF 2.0 scene.
    """
    overall_start = time.time()
    t0 = time.time()
    scene = _load_with_sparse_parts_connectivity(path)

    # We support only a single scene in glTF file.
    # Multiple scenes are rarely used and it was even proposed to remove them from the glTF 2.0 specification.
    # See https://github.com/KhronosGroup/glTF/issues/1542
    if len(scene.scenes) > 1:
        raise ValueError("Only one scene in the glTF file is supported.")

    # Please use .glb, we will handle .gltf with an ugly trick
    if path.endswith(".gltf"):
        t_convert = time.time()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            scene.save_binary(temp_file.name)
            temp_file_path = temp_file.name
        t_reload = time.time()
        scene = _load_with_sparse_parts_connectivity(temp_file_path)
        os.remove(temp_file_path)

    t_scene = time.time()
    gltf = gltfScene(scene, annotated=annotated)

    # Load the segmentation and articulation annotations
    if stk_segmentation is not None:
        t_seg = time.time()
        gltf.load_stk_segmentation(stk_segmentation)
    if stk_articulation is not None:
        if stk_segmentation is None:
            raise ValueError("Please provide the segmentation annotations as well.")
        t_art = time.time()
        gltf.load_stk_articulation(stk_articulation)
    if stk_precomputed_segmentation is not None:
        t_pre = time.time()
        gltf.load_stk_precomputed_segmentation(stk_precomputed_segmentation)

    return gltf
