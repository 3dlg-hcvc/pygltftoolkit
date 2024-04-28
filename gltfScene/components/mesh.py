import json


class Mesh():
    def __init__(self, id: int, primitives: list, name: str = None):
        """
        Initialize the Mesh object
        Args:
            id: int, the id of the mesh
            name: str, the name of the mesh
            primitives: list, the primitives of the mesh
        Properties:
            id: int, the id of the mesh
            name: str, the name of the mesh
            primitives: list, the primitives of the mesh
        """
        self.id: int = id

        self.name: str = None
        if name is not None:
            self.name: str = name

        self.primitives: list = primitives

    def __str__(self) -> str:
        class_dict = self.__dict__()
        return f"Mesh: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"id": self.id,
                      "name": self.name,
                      "primitives": [primitive.__dict__() for primitive in self.primitives]}
        return class_dict
