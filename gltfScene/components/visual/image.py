import json

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

        # Check if power of two and resize if not
        """if not (image.width & (image.width - 1) == 0) or not (image.height & (image.height - 1) == 0):
            new_width = 2 ** (image.width - 1).bit_length()
            new_height = 2 ** (image.height - 1).bit_length()
            image = image.resize((new_width, new_height))"""
        self.image: Image = image
        self.name: str = name
        self.mimeType: str = mimeType
        self.width: int = image.width
        self.height: int = image.height



    def __str__(self) -> str:
        class_dict = {"name": self.name,
                      "mimeType": self.mimeType,
                      "width": self.width,
                      "height": self.height}
        return f"TextureImage: {json.dumps(class_dict)}"

    def __dict__(self) -> dict:
        class_dict = {"name": self.name,
                      "mimeType": self.mimeType,
                      "width": self.width,
                      "height": self.height}
        return class_dict
