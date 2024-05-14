import json

import numpy as np

from .mesh import Mesh


class Node():
    def __init__(self, id: int,
                 children: list,
                 mesh: Mesh,
                 matrix: list = np.asarray([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], dtype=np.float32),
                 translation: list = [0, 0, 0],
                 rotation: list = [0, 0, 0, 1],
                 scale: list = [1, 1, 1],
                 name: str = None):
        """
        Initialize the Node object, initialize next components recursively
        Args:
            id: int, the id of the node
            children: list, the children of the node
            mesh: pygltflib.gltfScene.components.Mesh, Mesh object referenced by the current node
            matrix: list or np.ndarray(np.float32), the matrix of the node
            translation: list or np.ndarray(np.float32), the translation of the node
            rotation: list or np.ndarray(np.float32), the rotation of the node
            scale: list or np.ndarray(np.float32), the scale of the node
            name: str, the name of the node
        Properties:
            id: int, the id of the node
            children: list, the children of the node
            mesh: pygltflib.gltfScene.components.Mesh, Mesh object referenced by the current node
            matrix: np.ndarray(np.float32), the matrix of the node
            translation: np.ndarray(np.float32), the translation of the node
            rotation: np.ndarray(np.float32), the rotation of the node
            scale: np.ndarray(np.float32), the scale of the node
            cameras: list, the cameras of the node
            name: str, the name of the node
        """
        self.id: int = id
        self.children: list = children
        self.mesh = None
        if mesh is not None:
            self.mesh = mesh

        if isinstance(matrix, list):
            matrix = np.asarray(matrix, dtype=np.float32)
        elif isinstance(matrix, np.ndarray) and matrix.dtype != np.float32:
            raise ValueError("Matrix must be of type np.float32 or list.")
        self.matrix: np.ndarray = matrix
        if isinstance(translation, list):
            translation = np.asarray(translation, dtype=np.float32)
        elif isinstance(translation, np.ndarray) and translation.dtype != np.float32:
            raise ValueError("Translation must be of type np.float32 or list.")
        self.translation: np.ndarray = translation
        if isinstance(rotation, list):
            rotation = np.asarray(rotation, dtype=np.float32)
        elif isinstance(rotation, np.ndarray) and rotation.dtype != np.float32:
            raise ValueError("Rotation must be of type np.float32 or list.")
        self.rotation: np.ndarray = rotation
        if isinstance(scale, list):
            scale = np.asarray(scale, dtype=np.float32)
        elif isinstance(scale, np.ndarray) and scale.dtype != np.float32:
            raise ValueError("Scale must be of type np.float32 or list.")
        self.scale: np.ndarray = scale

        self.cameras = []  # TODO initialize and support cameras

        self.name: str = None
        if name is not None:
            self.name = name

    def transform_coordinate_frame(self, original_coordinate_frame_4x4: np.ndarray, target_coordinate_frame_4x4: np.ndarray):
        """
        Transform the coordinate frame of the node and its children.
        Args:
            original_coordinate_frame: np.ndarray, the original coordinate frame (4x4)
            target_coordinate_frame: np.ndarray, the target coordinate frame (4x4)
        """
        if self.matrix is not None:
            self.matrix = np.dot(np.dot(self.matrix, original_coordinate_frame_4x4.T), target_coordinate_frame_4x4)

        if self.translation is not None:
            translation_homogeneous = np.append(self.translation, 1)
            translation_transformed = np.dot(np.dot(translation_homogeneous, original_coordinate_frame_4x4.T),
                                             target_coordinate_frame_4x4)
            self.translation = translation_transformed[:3]

        for child in self.children:
            child.transform_coordinate_frame(original_coordinate_frame_4x4, target_coordinate_frame_4x4)

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"Node: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"id": self.id,
                      "len(children)": len(self.children),
                      "children": [child.__dict__() for child in self.children],
                      "mesh": self.mesh.__dict__() if self.mesh is not None else None,
                      "matrix": self.matrix.tolist() if self.matrix is not None else None,
                      "translation": self.translation.tolist() if self.translation is not None else None,
                      "rotation": self.rotation.tolist() if self.rotation is not None else None,
                      "scale": self.scale.tolist() if self.scale is not None else None,
                      "name": self.name}
        return class_dict
