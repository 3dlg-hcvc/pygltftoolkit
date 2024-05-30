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
