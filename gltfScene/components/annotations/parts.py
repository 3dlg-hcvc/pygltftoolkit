import json

import numpy as np


class TriSegment():
    def __init__(self, meshIndex: int, triIndex: list, segIndex: int = None):
        """
        Initialize the TriSegment object
        Args:
            meshIndex: int, glTF mesh index
            triIndex: list, segment information
            segIndex: int, precomputed segment index
        Properties:
            meshIndex: int, glTF mesh index
            triIndex: list, segment information
            segIndex: int, precomputed segment index
        """
        self.meshIndex: int = meshIndex
        self.triIndex: list = triIndex
        self.segIndex: int = segIndex

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"TriSegment: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"meshIndex": self.meshIndex,
                      "triIndex": self.triIndex,
                      "segIndex": self.segIndex}
        return class_dict


class SegmentationPart():
    def __init__(self, pid: int, name: str, label: str, trisegments: list):
        """
        Initialize the SegmentationPart object
        Args:
            pid: int, the id of the part
            name: str, the name of the part
            label: str, the label of the part
            trisegments: list(TriSegment), the list of segments
        Properties:
            pid: int, the id of the part
            name: str, the name of the part
            label: str, the label of the part
            trisegments: list(TriSegment), the list of segments
        """
        self.pid: int = pid
        self.name: str = name
        self.label: str = label
        self.trisegments: list = trisegments

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"SegmentationPart: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"pid": self.pid,
                      "name": self.name,
                      "label": self.label,
                      "trisegments": [trisegment.__dict__() for trisegment in self.trisegments]}
        return class_dict


class ArticulatedPart():
    def __init__(self, pid: int, type: str, origin: np.ndarray, axis: np.ndarray):
        """
        Initialize the ArticulatedPart object
        Args:
            pid: id, the id of the part
            type: str, the type of the part
            origin: np.ndarray, the origin of the part
            axis: np.ndarray, the axis of the part
        Properties:
            pid: id, the id of the part
            type: str, the type of the part
            origin: np.ndarray, the origin of the part
            axis: np.ndarray, the axis of the part
        """
        self.pid: int = pid

        self.type: str = type
        self.origin: np.ndarray = origin
        self.axis: np.ndarray = axis

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"ArticulatedPart: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"pid": self.pid,
                      "type": self.type,
                      "origin": self.origin.tolist(),
                      "axis": self.axis.tolist()}
        return class_dict

class PrecomputedPart():
    def __init__(self, segIndex: int, trisegments: list):
        """
        Initialize the PrecomputedPart object
        Args:
            segIndex: int, the segmentation index of the part
            trisegments: list(TriSegment), the list of segments
        Properties:
            segIndex: int, the segmentation index of the part
            trisegments: list(TriSegment), the list of segments
        """
        self.segIndex: int = segIndex
        self.trisegments: list = trisegments

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"PrecomputedPart: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"segIndex": self.segIndex,
                      "trisegments": [trisegment.__dict__() for trisegment in self.trisegments]}
        return class_dict
