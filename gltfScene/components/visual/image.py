from PIL import Image


class TextureImage():
    def __init__(self, image: Image, mimeType: str, name: str = None) -> None:
        """
        Initialize the TextureImage object
        Args:
            image: PIL.Image, the image
            mimeType: str, the mimeType of the image. Either of ["image/jpeg", "image/png"]
            name: str, the name of the image
        Properties:
            image: PIL.Image, the image
            name: str, the name of the image
            mimeType: str, the mimeType of the image. Either of ["image/jpeg", "image/png"]
            width: int, the width of the image
            height: int, the height of the image
        """
        self.image: Image = image
        self.name: str = name
        self.mimeType: str = mimeType
        self.width: int = image.width
        self.height: int = image.height
