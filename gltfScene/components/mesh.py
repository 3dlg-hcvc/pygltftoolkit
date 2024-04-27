
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
