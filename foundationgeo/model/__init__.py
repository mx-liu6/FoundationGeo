import importlib
from typing import *

if TYPE_CHECKING:
    from .base import FoundationGeo as FoundationGeoBase
    from .v1 import FoundationGeo as FoundationGeoV1


def import_model_class_by_version(version: str) -> Union[Type['FoundationGeoBase'], Type['FoundationGeoV1']]:
    if version not in {'base', 'v1'}:
        raise ValueError(f'Unsupported model version: {version}. Available versions: "base", "v1".')

    try:
        module = importlib.import_module(f'.{version}', __package__)
    except ModuleNotFoundError:
        raise ValueError(f'Model version "{version}" not found.')

    if not hasattr(module, 'FoundationGeo'):
        raise ValueError(f'Class \"FoundationGeo\" not found in module {module.__name__}.')

    return getattr(module, 'FoundationGeo')
