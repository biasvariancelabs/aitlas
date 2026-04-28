import importlib.metadata


try:
    __version__ = importlib.metadata.version("aitlas")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
