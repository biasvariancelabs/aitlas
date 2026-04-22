class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        def _register(cls):
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} is already registered in {self._name}")
            self._module_dict[key] = cls
            return cls

        return _register

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(
                f"'{name}' is not found in {self._name}. Available: {list(self._module_dict.keys())}"
            )
        return self._module_dict[name]

    def list_registered(self):
        """Returns a sorted list of all registered keys."""
        return sorted(list(self._module_dict.keys()))

    def __str__(self):
        return f"{self._name} Registry: {', '.join(self.list_registered())}"


# Instantiate the global registries
BACKBONE_REGISTRY = Registry("Backbone")
NECK_REGISTRY = Registry("Neck")
DECODER_REGISTRY = Registry("Decoder")
HEAD_REGISTRY = Registry("Head")
ADAPTER_REGISTRY = Registry("Adapter")
