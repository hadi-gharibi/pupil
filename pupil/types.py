from typing import Any, Dict, NewType, Protocol

from nptyping import NDArray

NDArray2D = NewType("NDArray2D", NDArray[(Any, Any), Any])  # type: ignore


class IsDataclass(Protocol):
    __dataclass_fields__: Dict
