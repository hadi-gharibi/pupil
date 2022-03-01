from dataclasses import dataclass


@dataclass
class FaissConf:
    nlist : int  = 100 # number of splits
    nprobe : int  = 5 # number of buckets to check