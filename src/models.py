from dataclasses import dataclass
from enum import Enum


class Orientation(Enum):
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZXY = 4
    ZYX = 5

@dataclass(frozen=True)
class Package:
    id: str
    width: int
    length: int
    height: int
    value: float
    weight: float
    is_fragile: bool
    eta: int

@dataclass(frozen=True)
class Truck:
    width: int
    length: int
    height: int
    max_capacity: float

@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int
    z: int

@dataclass
class Placement:
    package_id: str
    position: Coordinate | None
    orientation: Orientation

@dataclass
class State:
    placements: dict[str, Placement]

@dataclass(frozen=True)
class Problem:
    truck: Truck
    packages: dict[str, Package]