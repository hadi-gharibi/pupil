from dataclasses import dataclass
from nptyping import NDArray
from typing import Any, NewType, Protocol, Dict

NDArray2D = NewType("NDArray2D", NDArray[(Any, Any), Any]) # type: ignore

class IsDataclass(Protocol):
    __dataclass_fields__: Dict

@dataclass
class FaissConf:
    nlist : int  = 100 # number of splits
    nprobe : int  = 5 # number of buckets to check