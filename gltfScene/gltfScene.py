import codecs
import copy
import io
import json
import struct
import tempfile
from typing import Tuple

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pygltflib
import trimesh
from PIL import Image
from pygltflib import GLTF2

from .components import Mesh, Node, Primitive
from .components.annotations import (
    ArticulatedPart,
    PrecomputedPart,
    SegmentationPart,
    TriSegment,
)
from .components.visual import PBRMaterial, Sampler, TextureImage, TextureMaterial


def lookat(eye, target, up):
    z = np.array(eye) - np.array(target)
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    view_matrix = np.array([
        [x[0], x[1], x[2], -np.dot(x, eye)],
        [y[0], y[1], y[2], -np.dot(y, eye)],
        [z[0], z[1], z[2], -np.dot(z, eye)],
        [0, 0, 0, 1]
    ])

    return view_matrix


def perspective(fov, aspect, near, far):
    f = 1.0 / np.tan(fov / 2.0)
    d = far - near

    projection_matrix = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / d, -2 * far * near / d],
        [0, 0, -1, 0]
    ])

    return projection_matrix


def rgb_to_hex_int(rgb):
    if any(not (0 <= val <= 1) for val in rgb):
        raise ValueError("RGB values should be in the range [0, 1]")
    r, g, b = (int(val * 255) for val in rgb)

    hex_int = (r << 16) + (g << 8) + b
    return hex_int


def _get_unpack_format(component_type: int) -> Tuple[str, int]:
    """
    Get the unpack format and the size of the component type.
    Args:
        component_type: int, the component type of the accessor (glTF 2.0)
    Returns:
        type_char: string, the unpack format of the component type
        unit_size: int, the size of the component type
    """
    if component_type == 5120:  # BYTE
        return "b", 1
    elif component_type == 5121:  # UNSIGNED_BYTE
        return "B", 1
    elif component_type == 5122:  # SHORT
        return "h", 2
    elif component_type == 5123:  # UNSIGNED_SHORT
        return "H", 2
    elif component_type == 5125:  # UNSIGNED_INT
        return "I", 4
    elif component_type == 5126:  # FLOAT
        return "f", 4
    else:
        raise ValueError(f"Unknown component type: {component_type}")


# Credit - Ivan Tam
def _get_num_components(type: str) -> int:
    """
    Get the number of components of the accessor type.
    Args:
        type: string, the type of the accessor (glTF 2.0)
    Returns:
        num_components: int, the number of components of the accessor type
    """
    if type == "SCALAR":
        return 1
    elif type == "VEC2":
        return 2
    elif type == "VEC3":
        return 3
    elif type == "VEC4":
        return 4
    elif type == "MAT2":
        return 4
    elif type == "MAT3":
        return 9
    elif type == "MAT4":
        return 16
    else:
        raise ValueError(f"Unknown type: {type}")


# Credit - Ivan Tam
def _read_buffer(scene: GLTF2, accessor_idx) -> list:
    """
    Read the data buffer pointed by the accessor.
    Args:
        scene: GLTF2, the glTF 2.0 scene
        accessor_idx: int, the index of the accessor
    Returns:
        results: list, the data pointed by the accessor
    """
    accessor = scene.accessors[accessor_idx]
    buffer_view = scene.bufferViews[accessor.bufferView]
    buffer = scene.buffers[buffer_view.buffer]
    data = scene.get_data_from_buffer_uri(buffer.uri)
    type_char, unit_size = _get_unpack_format(accessor.componentType)
    num_components = _get_num_components(accessor.type)
    unpack_format = f"<{type_char * num_components}"
    data_size = unit_size * num_components
    results = []
    for i in range(accessor.count):
        idx = buffer_view.byteOffset + accessor.byteOffset + i * data_size
        binary_data = data[idx:idx+data_size]
        result = struct.unpack(unpack_format, binary_data)
        results.append(result[0] if num_components == 1 else result)
    return results


def quaternion_to_rotation_matrix(q):
    """
    Transform quaternion to rotation matrix
    Args:
        q: np.ndarray, quaternion
    Returns:
        rotation_matrix: np.ndarray, rotation matrix
    """
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w, 0],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w, 0],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2, 0],
        [0,                       0,                   0,             1]
    ])


class gltfScene():
    def __init__(self, gltf2: GLTF2):
        """
        Initialize the glTF scene object
        Args:
            gltf2: GLTF2, the glTF 2.0 scene
        Properties:
            nodes: list, the nodes in the glTF 2.0 scene
            faces: numpy.ndarray(np.int_), the faces of the mesh
            vertices: numpy.ndarray(np.float32), the vertices of the mesh with transformation applied
            normals: numpy.ndarray(np.float32), the normals of the mesh
            no_transform_vertices: numpy.ndarray(np.float32), the vertices of the mesh without transformation applied
            node_map: numpy.ndarray(np.int_), the node map of the vertices
            mesh_map: numpy.ndarray(np.int_), the mesh map of the vertices
            primitive_map: numpy.ndarray(np.int_), the primitive map of the vertices.
                           Note that primitive_map is relative with respect for each mesh as primitives do not have global index
            has_segmentation: bool, whether the scene has segmentation annotations
            has_articulation: bool, whether the scene has articulation annotations
            has_precomputed_segmentation: bool, whether the scene has precomputed segmentation annotations
            segmentation_parts: dict(pygltftoolkit.gltfScene/components.annotations.segmentationPart), the segmentation parts of the scene
            articulation_parts: dict, the articulation parts of the scene
            precomputed_segmentation_parts: dict, the precomputed segmentation parts of the scene
            segmentation_map: numpy.ndarray(np.int_), the segmentation map of the vertices
            precomputed_segmentation_map: numpy.ndarray(np.int_), the precomputed segmentation map of the vertices
        """

        self.gltf2: GLTF2 = gltf2
        self.nodes: list = []
        self.samplers: list = []
        self.faces: np.ndarray = np.empty((0, 3), dtype=np.int_)
        self.vertices: np.ndarray = np.empty((0, 3), dtype=np.float32)  # With transformation applied
        self.no_transform_vertices: np.ndarray = np.empty((0, 3), dtype=np.float32)  # Without transformation applied
        self.normals: np.ndarray = np.empty((0, 3), dtype=np.float32)

        self.node_map: np.ndarray = np.empty((0), dtype=np.int_)
        self.node_lookup: dict = {}

        self.mesh_map: np.ndarray = np.empty((0), dtype=np.int_)

        self.primitive_map: np.ndarray = np.empty((0), dtype=np.int_)

        self._global_vertex_counter: int = 0

        self.has_segmentation: bool = False
        self.has_articulation: bool = False
        self.has_precomputed_segmentation: bool = False

        self.segmentation_parts: dict = {}
        self.articulation_parts: dict = {}
        self.precomputed_segmentation_parts: dict = {}

        self.segmentation_map: np.ndarray = np.empty((0), dtype=np.int_)
        self.precomputed_segmentation_map: np.ndarray = np.empty((0), dtype=np.int_)

        for pygltflib_sampler in self.gltf2.samplers:
            magFilter = pygltflib_sampler.magFilter
            minFilter = pygltflib_sampler.minFilter
            wrapS = pygltflib_sampler.wrapS
            wrapT = pygltflib_sampler.wrapT
            new_sampler = Sampler(magFilter=magFilter, minFilter=minFilter, wrapS=wrapS, wrapT=wrapT)
            self.samplers.append(new_sampler)

        for node_id in self.gltf2.scenes[0].nodes:
            new_node = self.initialize_node(node_id)
            self.node_lookup[node_id] = {"node": new_node, "parent_id": None}
            self.nodes.append(new_node)

    def initialize_node(self, node_id: int,
                        parent_transform: np.ndarray = np.asarray([[1, 0, 0, 0],
                                                                   [0, 1, 0, 0],
                                                                   [0, 0, 1, 0],
                                                                   [0, 0, 0, 1]], dtype=np.float32)) -> Node:
        """
        Initialize the node object and its children.
        Args:
            node_id: int, the id of the node
            parent_transform: np.ndarray, the transformation matrix of the parent node
        """
        checked_attributes = ["COLOR_0", "NORMAL", "POSITION", "TEXCOORD_0"]
        parent_transform = parent_transform.astype(dtype=np.float32)
        pygltflib_node = self.gltf2.nodes[node_id]
        if pygltflib_node.matrix is not None:
            if len(pygltflib_node.matrix) == 16:
                matrix = np.array(pygltflib_node.matrix, dtype=np.float32).reshape(4, 4)
            else:
                matrix = np.asarray(pygltflib_node.matrix, np.float32)
            new_parent_transform = np.dot(parent_transform, matrix)
        else:
            translation = np.asarray(pygltflib_node.translation
                                     if pygltflib_node.translation is not None else [0, 0, 0])
            rotation = np.asarray(pygltflib_node.rotation
                                  if pygltflib_node.rotation is not None else [0, 0, 0, 1])
            scale = np.asarray(pygltflib_node.scale
                               if pygltflib_node.scale is not None else [1, 1, 1])
            scale_matrix = np.diag([scale[0], scale[1], scale[2], 1])
            rotation_matrix = quaternion_to_rotation_matrix(rotation)
            translation_matrix = np.array([
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1]
            ])
            transform = np.dot(translation_matrix, np.dot(rotation_matrix, scale_matrix))
            new_parent_transform = np.dot(parent_transform, transform).astype(dtype=np.float32)

        children = []
        for child_id in pygltflib_node.children:
            new_child = self.initialize_node(child_id, parent_transform=new_parent_transform)
            self.node_lookup[child_id] = {"node": new_child, "parent_id": node_id}
            children.append(new_child)

        if pygltflib_node.mesh is not None:
            node_vertices = np.empty((0, 3), dtype=np.float32)

            pygltflib_mesh = self.gltf2.meshes[pygltflib_node.mesh]
            primitives = []
            for primitive_id, pygltflib_primitive in enumerate(pygltflib_mesh.primitives):
                attributes = {}
                indices_accessor_idx = pygltflib_primitive.indices
                if indices_accessor_idx is not None:
                    indices_data = np.asarray(_read_buffer(self.gltf2, indices_accessor_idx))
                    indices_data += self._global_vertex_counter
                    indices = indices_data.reshape(-1, 3)
                else:
                    # Triangle soup mode
                    accessor_id = getattr(pygltflib_primitive.attributes, "POSITION")
                    temp_attr = _read_buffer(self.gltf2, accessor_id)
                    vertices = np.asarray(temp_attr, dtype=np.float32)
                    indices = np.arange(self._global_vertex_counter, self._global_vertex_counter + len(vertices)).reshape(-1, 3)
                self.faces = np.vstack((self.faces, indices))
                self.node_map = np.concatenate((self.node_map,
                                                np.array([node_id] * len(indices), dtype=np.int_)))
                self.mesh_map = np.concatenate((self.mesh_map,
                                                np.array([pygltflib_node.mesh] * len(indices), dtype=np.int_)))
                self.primitive_map = np.concatenate((self.primitive_map,
                                                     np.array([primitive_id] * len(indices), dtype=np.int_)))
                for attribute_name in checked_attributes:
                    if getattr(pygltflib_primitive.attributes, attribute_name) is None:
                        continue
                    accessor_id = getattr(pygltflib_primitive.attributes, attribute_name)
                    temp_attr = _read_buffer(self.gltf2, accessor_id)
                    attributes[attribute_name] = np.asarray(temp_attr, dtype=np.float32)
                    if attribute_name == "POSITION":
                        position = attributes[attribute_name]
                        node_vertices = np.vstack((node_vertices, position))
                        self.no_transform_vertices = np.vstack((self.no_transform_vertices, position))
                        self._global_vertex_counter += len(position)
                    elif attribute_name == "NORMAL":
                        self.normals = np.vstack((self.normals, attributes[attribute_name]))
                if "COLOR_0" in attributes:
                    colors_data = attributes["COLOR_0"]
                    new_primitive = Primitive(
                        attributes=attributes,
                        vertex_colors=colors_data
                    )
                elif "TEXCOORD_0" in attributes:
                    material = self.gltf2.materials[pygltflib_primitive.material]
                    if material.pbrMetallicRoughness.baseColorTexture is not None:
                        texture = self.gltf2.textures[material.pbrMetallicRoughness.baseColorTexture.index]
                        image = self.gltf2.images[texture.source]
                        bufferView = self.gltf2.bufferViews[image.bufferView]
                        data = copy.deepcopy(self.gltf2).binary_blob()

                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            # Adopted from https://gitlab.com/dodgyville/pygltflib/-/blob/v1.16.2/pygltflib/__init__.py?ref_type=tags#L793
                            temp_file.write(data[bufferView.byteOffset:bufferView.byteOffset + bufferView.byteLength])
                            temp_file_path = temp_file.name

                        with Image.open(temp_file_path) as pil_image:
                            width, height = pil_image.size
                            image_data = pil_image.convert("RGBA")

                        texcoords = attributes["TEXCOORD_0"]
                        texture_image = TextureImage(image=image_data, mimeType=image.mimeType, name=image.name)
                        texture_material = TextureMaterial(image=texture_image, uv=texcoords, sampler=texture.sampler)
                        new_primitive = Primitive(
                            attributes=attributes,
                            material=texture_material
                        )
                    else:
                        new_primitive = Primitive(attributes=attributes)
                        # TODO handle properly and add default material
                else:
                    material = self.gltf2.materials[pygltflib_primitive.material]
                    baseColorFactor = material.pbrMetallicRoughness.baseColorFactor
                    metallicFactor = material.pbrMetallicRoughness.metallicFactor
                    roughnessFactor = material.pbrMetallicRoughness.roughnessFactor
                    new_primitive = Primitive(
                        attributes=attributes,
                        material=PBRMaterial(baseColorFactor=baseColorFactor,
                                             metallicFactor=metallicFactor,
                                             roughnessFactor=roughnessFactor)
                    )
                primitives.append(new_primitive)

            new_mesh = Mesh(
                id=pygltflib_node.mesh,
                name=pygltflib_mesh.name,
                primitives=primitives
            )
            if node_vertices.shape[1] == 3:
                ones = np.ones((node_vertices.shape[0], 1))
                node_vertices_homogeneous = np.hstack((node_vertices, ones))
            else:
                node_vertices_homogeneous = node_vertices
            transformed_vertices = np.dot(node_vertices_homogeneous, new_parent_transform.T)
            if node_vertices.shape[1] == 3:
                transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3][:, np.newaxis]
            self.vertices = np.vstack((self.vertices, transformed_vertices))
        else:
            new_mesh = None
        new_node = Node(
            id=node_id,
            children=children,
            mesh=new_mesh,
            matrix=new_parent_transform,
            translation=pygltflib_node.translation,
            rotation=pygltflib_node.rotation,
            scale=pygltflib_node.scale,
            name=pygltflib_node.name
        )
        return new_node

    def load_stk_segmentation(self, stk_segmentation: str):
        """
        Load the segmentation annotations produced by the STK.
        Args:
            stk_segmentation: string, the path to the segmentation annotations produced by the STK
        """
        self.has_segmentation = True
        self.segmentation_map = np.empty((len(self.node_map)), dtype=np.int_)

        with open(stk_segmentation, "r") as f:
            stk_segmentation = json.load(f)
        for part in stk_segmentation["parts"]:
            if part is not None:
                pid = int(part["pid"])
                label = part["label"]
                trisegments = []
                for seg_data in part["partInfo"]["meshTri"]:
                    seg_mesh_index = int(seg_data["meshIndex"])
                    current_mesh = np.where(self.mesh_map == seg_mesh_index)[0]
                    for tri_data in seg_data["triIndex"]:
                        if type(tri_data) is int:
                            self.segmentation_map[current_mesh[tri_data]] = pid
                        elif type(tri_data) is list:
                            self.segmentation_map[current_mesh[tri_data[0]:tri_data[1]]] = pid
                    segIndex = None
                    if "segIndex" in seg_data:
                        segIndex = seg_data["segIndex"]
                    trisegment = TriSegment(meshIndex=seg_mesh_index, triIndex=seg_data["triIndex"], segIndex=segIndex)
                    trisegments.append(trisegment)
                new_part = SegmentationPart(pid=pid, name=part["name"], label=label, trisegments=trisegments)
                self.segmentation_parts[pid] = new_part

    def load_stk_segmentation_openable(self, stk_segmentation: str):
        """
        Load the segmentation annotations produced by the STK. Loads parts for "openable" tasks, e.g. S2O.
        That is, involves additional annotation processing (e.g. combining handles with doors, etc.)
        Args:
            stk_segmentation: string, the path to the segmentation annotations produced by the STK
        """
        self.has_segmentation = True
        self.segmentation_map = np.empty((len(self.node_map)), dtype=np.int_)
        with open(stk_segmentation, "r") as f:
            stk_segmentation = json.load(f)
        annotated_parts = {}
        base_part_id = None
        for part in stk_segmentation["parts"]:
            if part:
                part_info = part["partInfo"]
                part_label = part_info["label"]
                pid = int(part_info["partId"])
                if part_label in ["drawer", "door", "lid"]:
                    annotated_parts[pid] = part_label
                elif part_label in ["bed", "bunk beds", "base"]:
                    annotated_parts[pid] = "base"
                    base_part_id = int(part_info["partId"])
        for part in stk_segmentation["parts"]:
            if part:
                part_info = part["partInfo"]
                part_label = part_info["label"]
                if part_label in ["bed", "bunk beds"]:
                    part_label = "base"
                pid = int(part_info["partId"])
                trisegments = []
                for seg_data in part_info["meshTri"]:
                    seg_mesh_index = int(seg_data["meshIndex"])
                    current_mesh = np.where(self.mesh_map == seg_mesh_index)[0]
                    for tri_data in seg_data["triIndex"]:
                        if type(tri_data) is int:
                            self.segmentation_map[current_mesh[tri_data]] = pid
                        elif type(tri_data) is list:
                            self.segmentation_map[current_mesh[tri_data[0]:tri_data[1]]] = pid
                    segIndex = None
                    if "segIndex" in seg_data:
                        segIndex = seg_data["segIndex"]
                    trisegment = TriSegment(meshIndex=seg_mesh_index, triIndex=seg_data["triIndex"], segIndex=segIndex)
                    trisegments.append(trisegment)
                if part_label in ["drawer", "door", "lid"]:
                    new_part = SegmentationPart(pid=pid, name=part["name"], label=part_label, trisegments=trisegments)
                    self.segmentation_parts[pid] = new_part
                else:
                    connected_ids = stk_segmentation["connectivityGraph"][pid]
                    connected_to_label = None
                    connected_id = None
                    if part_label not in ["pillow", "quilt"]:
                        for c_id in connected_ids:
                            if c_id in annotated_parts.keys():
                                connected_to_label = annotated_parts[c_id]
                                connected_id = int(c_id)
                            else:
                                connected_to_label = "base"
                                connected_id = int(c_id)
                    if not connected_id or connected_to_label == "base" or part_label in ["pillow", "quilt", "base"]:
                        if base_part_id in self.segmentation_parts:
                            self.segmentation_parts[base_part_id].trisegments.extend(trisegments)
                        else:
                            new_part = SegmentationPart(pid=base_part_id, name="base", label="base", trisegments=trisegments)
                            self.segmentation_parts[base_part_id] = new_part
                        self.segmentation_map[self.segmentation_map == pid] = base_part_id
                    else:
                        if connected_id in self.segmentation_parts:
                            self.segmentation_parts[connected_id].trisegments.extend(trisegments)
                        else:
                            new_part = SegmentationPart(pid=connected_id, name=annotated_parts[connected_id], label=connected_to_label, trisegments=trisegments)
                            self.segmentation_parts[connected_id] = new_part
                        self.segmentation_map[self.segmentation_map == pid] = connected_id

    def load_stk_articulation(self, stk_articulation: str):
        """
        Load the articulation annotations produced by the STK.
        Args:
            stk_articulation: string, the path to the articulation annotations produced by the STK
        """
        if not self.has_segmentation:
            raise ValueError("Segmentation annotations must be loaded before loading articulation annotations.")

        self.has_articulation = True

        with open(stk_articulation, "r") as f:
            try:
                stk_articulation = json.load(f)
            except json.JSONDecodeError:
                try:
                    stk_articulation = json.load(codecs.open(stk_articulation, 'r', 'utf-8-sig'))
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format.")
        for articulation in stk_articulation["annotation"]["articulations"]:
            pid = articulation["pid"]
            type = articulation["type"]
            origin = np.asarray(articulation["origin"])
            axis = np.asarray(articulation["axis"])
            new_part = ArticulatedPart(pid=pid,
                                       type=type,
                                       origin=origin,
                                       axis=axis)
            self.articulation_parts[pid] = new_part

    def load_stk_precomputed_segmentation(self, stk_precomputed_segmentation: str):
        """
        Load the precomputed segmentation annotations produced by the STK.
        Args:
            stk_precomputed_segmentation: string, the path to the precomputed segmentation annotations produced by
            the STK
        """
        self.has_precomputed_segmentation = True
        self.precomputed_segmentation_map = np.empty((len(self.node_map)), dtype=np.int_)
        with open(stk_precomputed_segmentation, "r") as f:
            stk_precomputed_segmentation = json.load(f)
        trisegments = []
        for segment in stk_precomputed_segmentation["segmentation"]:
            mesh_id = segment["meshIndex"]
            seg_id = segment["segIndex"]
            current_mesh = np.where(self.mesh_map == mesh_id)[0]

            for tri_data in segment["triIndex"]:
                if type(tri_data) is int:
                    self.precomputed_segmentation_map[current_mesh[tri_data]] = seg_id
                elif type(tri_data) is list:
                    self.precomputed_segmentation_map[current_mesh[tri_data[0]:tri_data[1]]] = seg_id

            trisegment = TriSegment(meshIndex=mesh_id, triIndex=tri_data, segIndex=seg_id)
            trisegments.append(trisegment)
            new_part = PrecomputedPart(seg_id, trisegments)
            self.precomputed_segmentation_parts[seg_id] = new_part

    def load_stk_precomputed_segmentation_flattened(self, stk_precomputed_segmentation: str):
        """
        Load the precomputed segmentation annotations produced by the STK without respecting the mesh boundaries.
        Args:
            stk_precomputed_segmentation: string, the path to the precomputed segmentation annotations produced by
            the STK
        """
        self.has_precomputed_segmentation = True
        self.precomputed_segmentation_map = np.empty((len(self.node_map)), dtype=np.int_)
        with open(stk_precomputed_segmentation, "r") as f:
            stk_precomputed_segmentation = json.load(f)
        trisegments = []
        for segment in stk_precomputed_segmentation["segmentation"]:
            seg_id = segment["segIndex"]

            for tri_data in segment["triIndex"]:
                if type(tri_data) is int:
                    self.precomputed_segmentation_map[tri_data] = seg_id
                elif type(tri_data) is list:
                    self.precomputed_segmentation_map[tri_data[0]:tri_data[1]] = seg_id

            trisegment = TriSegment(meshIndex=-1, triIndex=tri_data, segIndex=seg_id)
            trisegments.append(trisegment)
            new_part = PrecomputedPart(seg_id, trisegments)
            self.precomputed_segmentation_parts[seg_id] = new_part

    def transform_coordinate_frame(self, original_coordinate_frame: np.ndarray, target_coordinate_frame: np.ndarray):
        """
        Transform the coordinate frame of the scene.
        Args:
            original_coordinate_frame: np.ndarray, the original coordinate frame (3x3)
            target_coordinate_frame: np.ndarray, the target coordinate frame (3x3)
        """
        original_coordinate_frame_4x4 = np.eye(4)
        original_coordinate_frame_4x4[:3, :3] = original_coordinate_frame
        target_coordinate_frame_4x4 = np.eye(4)
        target_coordinate_frame_4x4[:3, :3] = target_coordinate_frame
        for node in self.nodes:
            node.transform_coordinate_frame(original_coordinate_frame_4x4, target_coordinate_frame_4x4)
        self.vertices = np.dot(self.vertices, original_coordinate_frame.T)
        self.vertices = np.dot(self.vertices, target_coordinate_frame)
        self.normals = np.dot(self.normals, original_coordinate_frame.T)
        self.normals = np.dot(self.normals, target_coordinate_frame)

    def rescale(self, scale: float):
        """
        Rescale the scene.
        Args:
            scale: float, the scale factor
        """
        self.vertices *= scale

    def color_faces(self, face_colors: np.ndarray, cut_primitives: bool = False):
        """
        Color the faces of the scene. Currently implemented for pygltflib.GLTF2 so can be exported.
        Args:
            face_colors: np.ndarray, the colors of the faces
            cut_primitives: bool, whether to cut the primitives
        """
        if len(face_colors) != len(self.faces):
            raise ValueError("Length of face colors must match the number of faces.")

        if face_colors.shape[1] != 4:
            face_colors = np.hstack((face_colors, np.ones((len(face_colors), 1), dtype=np.float32)))

        # Currently not implemented, however in the future it will be possible to assign colors not as COLOR_0 (vertex colors)
        # but as face colors using PBRMaterial.
        """if not cut_primitives:
            for primitive in np.unique(self.primitive_map):
                primitive_mask = self.primitive_map == primitive
                primitive_colors = face_colors[primitive_mask]
                if len(np.unique(primitive_colors)):
                    raise ValueError("Face colors must be unique for each primitive or you should allow primitive cutting.")"""

        def dfs_recolor(node, face_colors):
            if node.mesh is not None:
                for local_id, primitive in enumerate(node.mesh.primitives):
                    cur_primitive_mask = np.zeros_like(self.primitive_map, dtype=np.bool_)
                    cur_node_mask = self.node_map == node.id
                    cur_primitive_mask[cur_node_mask] = self.primitive_map[cur_node_mask] == local_id
                    new_face_colors = face_colors[cur_primitive_mask]
                    primitive.remove_visuals()
                    new_vertex_colors = np.zeros((len(self.vertices), 4), dtype=np.float32)
                    primitive_vertex_mask = np.zeros((len(self.vertices)), dtype=np.bool_)
                    for i, face_color in enumerate(new_face_colors):
                        new_vertex_colors[self.faces[cur_primitive_mask][i]] = face_color
                        primitive_vertex_mask[self.faces[cur_primitive_mask][i]] = True
                    new_vertex_colors = new_vertex_colors[primitive_vertex_mask]
                    primitive.color_vertices(new_vertex_colors)
            for child in node.children:
                dfs_recolor(child, face_colors)

        for node in self.nodes:
            dfs_recolor(node, face_colors)
        self._recolor_gltf2(face_colors)

    def _recolor_gltf2(self, face_colors: np.ndarray):
        """
        Recolor the faces of the scene for pygltflib.GLTF2.
        Args:
            face_colors: np.ndarray, the colors of the faces
        """
        # NOTE: This function is messy and will be updated. It does not remove existing materials from the buffer.
        blobs = [self.gltf2.binary_blob()]
        buffer_offset = len(blobs[0])

        self.gltf2.materials = []

        for mesh_index, mesh in enumerate(self.gltf2.meshes):
            for primitive_index, primitive in enumerate(mesh.primitives):
                primitive_mask = (self.primitive_map == primitive_index) & (self.mesh_map == mesh_index)
                primitive_faces = self.faces[primitive_mask]
                primitive_colors = face_colors[primitive_mask]

                primitive.material = None
                if primitive.attributes.COLOR_0 is not None:
                    primitive.attributes.COLOR_0 = None
                if primitive.attributes.TEXCOORD_0 is not None:
                    primitive.attributes.TEXCOORD_0 = None

                # We need defualt material for the primitive with colored vertices, according to GLTF 2.0 specs
                material = pygltflib.Material(pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorFactor=[1.0, 1.0, 1.0, 1.0]))
                material_index = len(self.gltf2.materials)
                self.gltf2.materials.append(material)

                primitive.material = material_index

                position_accessor = self.gltf2.accessors[primitive.attributes.POSITION]
                vertex_count = position_accessor.count
                vertex_colors = np.zeros((vertex_count, 4), dtype=np.float32)
                offset = -1
                for face, face_color in zip(primitive_faces, primitive_colors):
                    if offset == -1:
                        offset = np.min(face)
                    vertex_colors[face - offset] = face_color

                byte_length = vertex_colors.nbytes
                new_buffer_view = pygltflib.BufferView(
                    buffer=0,
                    byteOffset=buffer_offset,
                    byteLength=byte_length
                )

                buffer_offset += byte_length
                buffer_view_index = len(self.gltf2.bufferViews)
                self.gltf2.bufferViews.append(new_buffer_view)

                color_blob = struct.pack(f"{len(vertex_colors) * 4}f", *vertex_colors.flatten().tolist())
                blobs.append(color_blob)

                color_accessor = pygltflib.Accessor(
                    bufferView=buffer_view_index,
                    componentType=pygltflib.FLOAT,
                    count=len(vertex_colors),
                    type=pygltflib.VEC4,
                    max=vertex_colors.max(axis=0).tolist(),
                    min=vertex_colors.min(axis=0).tolist()
                )
                accessor_index = len(self.gltf2.accessors)
                self.gltf2.accessors.append(color_accessor)

                primitive.attributes.COLOR_0 = accessor_index

        self.gltf2.set_binary_blob(b"".join(blobs))

    def create_colored_trimesh(self, face_colors: np.ndarray):
        """
        Create a colored trimesh from the scene.
        Args:
            face_colors: np.ndarray, the colors of the faces
        """
        trimesh_faces = []
        trimesh_vertices = []

        for i, color in enumerate(face_colors):
            trimesh_faces.append([i*3, i*3+1, i*3+2])
            trimesh_vertices.append(self.vertices[self.faces[i]][0])
            trimesh_vertices.append(self.vertices[self.faces[i]][1])
            trimesh_vertices.append(self.vertices[self.faces[i]][2])
        mesh = trimesh.Trimesh(vertices=np.array(trimesh_vertices), faces=np.array(trimesh_faces))
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh, face_colors=face_colors * 255)
        return mesh


    def create_trimesh(self):
        """
        Create a trimesh from the scene with all visuals (textures and colors).
        """
        geometry_dict = {}

        def dfs_populate(node):
            nonlocal geometry_dict
            if node.mesh is not None:
                mesh_id = node.mesh.id
                for primitive_id, primitive in enumerate(node.mesh.primitives):
                    primitive_faces = self.faces[(self.primitive_map == primitive_id) & (self.mesh_map == mesh_id)]
                    unique_faces = np.unique(primitive_faces)
                    primitive_vertices = self.vertices[unique_faces]
                    new_primitive_faces = np.zeros_like(primitive_faces)
                    for face_idx, primitive_face in enumerate(primitive_faces):
                        for vert_idx, primitive_vertex in enumerate(primitive_face):
                            new_primitive_faces[face_idx][vert_idx] = np.where(unique_faces == primitive_vertex)[0]
                    if primitive.has_texture:
                        texture_image = primitive.material.texture
                        texture_uv = primitive.material.uv
                        if texture_image is not None:
                            image = texture_image.image
                            uv = texture_uv
                            visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
                            geometry_dict[f"{mesh_id}-{primitive_id}"] = trimesh.Trimesh(vertices=primitive_vertices, faces=new_primitive_faces, visual=visual)
                        else:
                            baseColorFactor = primitive.material.baseColorFactor
                            face_colors = np.ones((len(new_primitive_faces), 4), dtype=np.float32)
                            face_colors[:] = baseColorFactor
                            geometry_dict[f"{mesh_id}-{primitive_id}"] = trimesh.Trimesh(vertices=primitive_vertices, faces=new_primitive_faces, visual=trimesh.visual.color.ColorVisuals(face_colors=face_colors * 255))
                    elif primitive.has_colors:
                        baseColorFactor = primitive.material.baseColorFactor
                        face_colors = np.ones((len(new_primitive_faces), 4), dtype=np.float32)
                        face_colors[:] = baseColorFactor
                        face_colors *= primitive.vertex_colors[new_primitive_faces]
                        geometry_dict[f"{mesh_id}-{primitive_id}"] = trimesh.Trimesh(vertices=primitive_vertices, faces=new_primitive_faces, visual=trimesh.visual.color.ColorVisuals(face_colors=face_colors * 255))
                    elif primitive.has_baseColorFactor:
                        baseColorFactor = primitive.material.baseColorFactor
                        face_colors = np.ones((len(new_primitive_faces), 4), dtype=np.float32)
                        face_colors[:] = baseColorFactor
                        geometry_dict[f"{mesh_id}-{primitive_id}"] = trimesh.Trimesh(vertices=primitive_vertices, faces=new_primitive_faces, visual=trimesh.visual.color.ColorVisuals(face_colors=face_colors * 255))
                    else:
                        geometry_dict[f"{mesh_id}-{primitive_id}"] = trimesh.Trimesh(vertices=primitive_vertices, faces=new_primitive_faces)
            for child in node.children:
                dfs_populate(child)
        for node in self.nodes:
            dfs_populate(node)

        trimesh_scene = trimesh.Scene(geometry=geometry_dict)
        return trimesh_scene

    def interactiveVisualizer(self) -> meshcat.Visualizer:
        """
        Create an interactive visualizer for the scene.
        """
        vis = meshcat.Visualizer()
        return vis

    def staticVisualizer(self, background=False, grid=False, axes=False) -> meshcat.Visualizer:
        """
        Create a static visualizer for the scene.
        Args:
            background: bool, whether to show the background
            grid: bool, whether to show the grid
            axes: bool, whether to show the axes
        """
        vis = meshcat.Visualizer()
        if not background:
            vis["/Background"].set_property("visible", False)
        if not grid:
            vis["/Grid"].set_property("visible", False)
        if not axes:
            vis["/Axes"].set_property("visible", False)
        vis.open()
        return vis

    def _prepare_vis_context(self, vis):
        def add_node_to_scene(node, parent_transform=np.eye(4)):
            node_name = node.name if node.name is not None else f"node_{node.id}"
            transform = node.matrix
            current_transform = np.dot(parent_transform, transform)
            if node.mesh is not None:
                mesh_name = node.mesh.name if node.mesh.name is not None else f"mesh_{node.mesh.id}"
                for primitive_id, primitive in enumerate(node.mesh.primitives):
                    primitive_name = f"primitive_{primitive_id}"
                    attributes = primitive.attributes
                    global_faces = self.faces[(self.primitive_map == primitive_id) & (self.mesh_map == node.mesh.id)]
                    unique_vertices, new_indices = np.unique(global_faces, return_inverse=True)
                    faces = new_indices.reshape(global_faces.shape).tolist()
                    local_vertices = self.vertices[unique_vertices]

                    # Convert local_vertices to homogeneous coordinates
                    ones = np.ones((local_vertices.shape[0], 1))
                    local_vertices_homogeneous = np.hstack((local_vertices, ones))

                    # Apply the transformation
                    transformed_vertices = np.dot(local_vertices_homogeneous, current_transform.T)
                    transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3][:, np.newaxis]

                    if primitive.has_colors:
                        colors = primitive.vertex_colors[:, :3]
                        meshcat_geometry = g.TriangularMeshGeometry(transformed_vertices.tolist(), faces, color=colors)
                        meshcat_material = g.MeshLambertMaterial(vertexColors=True)
                        vis[node_name][mesh_name][primitive_name].set_object(g.Mesh(meshcat_geometry, meshcat_material))
                    elif primitive.has_texture:
                        texcoords = attributes["TEXCOORD_0"]
                        texture_image = primitive.material.texture.image
                        # Flip image
                        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
                        sampler_id = primitive.material.sampler
                        if sampler_id is not None:
                            sampler = self.samplers[sampler_id]
                            wrap = [sampler.wrapS, sampler.wrapT]
                            repeat = [1, 1]
                        else:
                            wrap = [1001, 1001]
                            repeat = [1, 1]
                        img_byte_arr = io.BytesIO()
                        texture_image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        png_image = g.PngImage(img_bytes)
                        texture = g.ImageTexture(image=png_image, wrap=wrap, repeat=repeat)
                        meshcat_material = g.MeshLambertMaterial(map=texture)
                        meshcat_geometry = g.TriangularMeshGeometry(transformed_vertices, faces, uvs=texcoords)
                        vis[node_name][mesh_name][primitive_name].set_object(g.Mesh(meshcat_geometry, meshcat_material))
                    else:
                        meshcat_geometry = g.TriangularMeshGeometry(transformed_vertices, faces)
                        baseColorFactor = primitive.material.baseColorFactor.astype(np.float32)
                        baseColorFactor = baseColorFactor[:3]
                        hex_color = rgb_to_hex_int(baseColorFactor)
                        meshcat_material = g.MeshLambertMaterial(color=hex(hex_color))
                        vis[node_name][mesh_name][primitive_name].set_object(g.Mesh(meshcat_geometry, meshcat_material))

            for child in node.children:
                add_node_to_scene(child, current_transform)

        for node in self.nodes:
            add_node_to_scene(node)

    def show(self, vis):
        """
        Render the GLTF scene using MeshCat.
        Args:
            vis: meshcat.Visualizer, the visualizer object
        """
        vis.delete()
        self._prepare_vis_context(vis)
        print("Press Ctrl+C to close the visualizer.")
        vis.open_with_listener()

    def render(self, vis, output_path, width=1024, height=512):
        """
        Render the GLTF scene using MeshCat.
        Args:
            vis: meshcat.Visualizer, the visualizer object
            output_path: str, the path to save the rendered images
        """
        vis.delete()
        self._prepare_vis_context(vis)
        image = vis.get_image(width, height)
        image.save(output_path, format='PNG')

    def export_gltf2(self, export_path):
        """
        Export self.gltf2 (pygltflib.GLTF2)
        Args:
            export_path: str, the path to export the glTF 2.0 file
        """
        self.gltf2.save(export_path)

    def __str__(self):
        class_dict = {"len(self.nodes)": len(self.nodes),
                      "nodes": [node.__dict__() for node in self.nodes],
                      "len(self.faces)": len(self.faces),
                      "len(self.vertices)": len(self.vertices),
                      "len(self.segmentation_parts)": len(self.segmentation_parts),
                      "segmentation_parts": [part.__dict__() for part in self.segmentation_parts.values()],
                      "len(self.articulation_parts)": len(self.articulation_parts),
                      "articulation_parts": [part.__dict__() for part in self.articulation_parts.values()],
                      "len(self.precomputed_segmentation_parts)": len(self.precomputed_segmentation_parts),
                      "precomputed_segmentation_parts":
                      [part.__dict__() for part in self.precomputed_segmentation_parts.values()]}
        return f"gltfScene: {json.dumps(class_dict)}"
