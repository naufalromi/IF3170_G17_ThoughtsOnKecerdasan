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

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Package id cannot be empty")
        if not all(
            type(value) is int for value in (self.width, self.length, self.height)
        ):
            raise TypeError("Package dimensions must be integers")
        if min(self.width, self.length, self.height) <= 0:
            raise ValueError("Package dimensions must be positive")
        if self.value < 0 or self.weight < 0:
            raise ValueError("Package value and weight cannot be negative")
        if type(self.is_fragile) is not bool:
            raise TypeError("Package is_fragile must be a boolean")
        if type(self.eta) is not int:
            raise TypeError("Package ETA must be an integer")
        if self.eta < 0:
            raise ValueError("Package ETA cannot be negative")


@dataclass(frozen=True)
class Truck:
    width: int
    length: int
    height: int
    max_capacity: float

    def __post_init__(self) -> None:
        if not all(
            type(value) is int for value in (self.width, self.length, self.height)
        ):
            raise TypeError("Truck dimensions must be integers")
        if min(self.width, self.length, self.height) <= 0:
            raise ValueError("Truck dimensions must be positive")
        if self.max_capacity < 0:
            raise ValueError("Truck capacity cannot be negative")


@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if not all(type(value) is int for value in (self.x, self.y, self.z)):
            raise TypeError("Coordinates must be integers")


@dataclass
class Placement:
    package_id: str
    position: Coordinate | None
    orientation: Orientation

    def __post_init__(self) -> None:
        if not self.package_id:
            raise ValueError("Placement package id cannot be empty")
        if not isinstance(self.orientation, Orientation):
            raise TypeError("Placement orientation must be an Orientation")


@dataclass
class State:
    placements: dict[str, Placement]


@dataclass(frozen=True)
class Problem:
    truck: Truck
    packages: dict[str, Package]

    def __post_init__(self) -> None:
        for package_id, package in self.packages.items():
            if package_id != package.id:
                raise ValueError(
                    f"Package key {package_id!r} does not match id {package.id!r}"
                )
