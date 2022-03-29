from enum import Enum, auto
from typing import Any, Dict, Iterable, NewType, Protocol, Sequence, TypeVar

from nptyping import NDArray

NDArray2D = NewType("NDArray2D", NDArray[(Any, Any), Any])  # type: ignore
T = TypeVar("T")


class IsDataclass(Protocol):
    __dataclass_fields__: Dict

class Distance(Enum):
    COSINE = auto()
    DOT_PRODUCT = auto()


SeqOfIndices = TypeVar("SeqOfIndices", bound= Sequence[int])
