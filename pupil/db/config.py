from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, NewType, Optional, Protocol

from nptyping import NDArray

NDArray2D = NewType("NDArray2D", NDArray[(Any, Any), Any])  # type: ignore


class IsDataclass(Protocol):
    __dataclass_fields__: Dict


class SimilarityType(Enum):
    COSINE = auto()
    DOT_PRODUCT = auto()


@dataclass
class FaissConf:
    nlist: Optional[int] = None  # number of splits
    nprobe: int = 5  # number of buckets to check
    similarity_type: SimilarityType = SimilarityType.COSINE
