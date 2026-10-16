import random
from collections.abc import Iterator
from copy import deepcopy
from itertools import chain

from src.models import Coordinate, Orientation, Package, Placement, Problem, State

def oriented_dimensions(
    package: Package,
    orientation: Orientation,
) -> tuple[int, int, int]:
    width = package.width
    length = package.length
    height = package.height

    if orientation == Orientation.XYZ:
        return width, length, height
    if orientation == Orientation.XZY:
        return width, height, length
    if orientation == Orientation.YXZ:
        return length, width, height
    if orientation == Orientation.YZX:
        return length, height, width
    if orientation == Orientation.ZXY:
        return height, width, length
    if orientation == Orientation.ZYX:
        return height, length, width

    raise ValueError(f"Unknown orientation: {orientation}")


def objective(problem: Problem, state: State) -> float:
    return sum(
        problem.packages[placement.package_id].value
        for placement in state.placements.values()
        if placement.position is not None
    )


def total_weight(problem: Problem, state: State) -> float:
    return sum(
        problem.packages[placement.package_id].weight
        for placement in state.placements.values()
        if placement.position is not None
    )


def is_valid(problem: Problem, state: State) -> bool:
    if not _is_well_formed(problem, state):
        return False

    return all(
        _is_inside_truck(problem, placement)
        for placement in state.placements.values()
    ) and (
        _has_no_collision(problem, state)
        and _is_within_capacity(problem, state)
        and _all_packages_supported(problem, state)
    )


def _is_well_formed(problem: Problem, state: State) -> bool:
    if set(state.placements) != set(problem.packages):
        return False

    for package_id, placement in state.placements.items():
        if not isinstance(placement, Placement):
            return False
        if placement.package_id != package_id:
            return False
        if not isinstance(placement.orientation, Orientation):
            return False
        if placement.position is not None and not isinstance(
            placement.position, Coordinate
        ):
            return False

    return True


def random_initial_state(problem: Problem, rng: random.Random) -> State:
    placements = {
        package_id: Placement(package_id, None, Orientation.XYZ)
        for package_id in problem.packages
    }
    package_ids = list(problem.packages)
    rng.shuffle(package_ids)

    for package_id in package_ids:
        if rng.random() < 0.5:
            continue

        package = problem.packages[package_id]
        orientation = rng.choice(list(Orientation))
        width, length, height = oriented_dimensions(package, orientation)
        truck = problem.truck

        if width > truck.width or length > truck.length or height > truck.height:
            continue

        placement = Placement(
            package_id,
            Coordinate(
                rng.randint(0, truck.width - width),
                rng.randint(0, truck.length - length),
                0,
            ),
            orientation,
        )
        placements[package_id] = placement
        state = State(placements)

        if not is_valid(problem, state):
            placements[package_id] = Placement(package_id, None, Orientation.XYZ)

    return State(placements)


def generate_valid_neighbor(
    problem: Problem,
    state: State,
    rng: random.Random,
    max_attempts: int = 100,
) -> State | None:
    if not state.placements:
        return deepcopy(state)

    moves = ["move", "rotate"]
    if len(state.placements) >= 2:
        moves.append("swap")

    for _ in range(max_attempts):
        move = rng.choice(moves)
        if move == "move":
            candidate = _move_neighbor(problem, state, rng)
        elif move == "rotate":
            candidate = _rotate_neighbor(state, rng)
        else:
            candidate = _swap_neighbor(state, rng)

        if (
            _state_key(candidate) != _state_key(state)
            and is_valid(problem, candidate)
        ):
            return candidate

    return None


def generate_valid_neighbors(problem: Problem, state: State) -> Iterator[State]:
    if not is_valid(problem, state):
        raise ValueError("Cannot generate neighbors from an invalid state")

    seen = {_state_key(state)}

    for package_id, placement in state.placements.items():
        package = problem.packages[package_id]
        width, length, height = oriented_dimensions(package, placement.orientation)
        truck = problem.truck

        if width <= truck.width and length <= truck.length and height <= truck.height:
            positions = chain(
                (None,),
                (
                    Coordinate(x, y, z)
                    for x in range(truck.width - width + 1)
                    for y in range(truck.length - length + 1)
                    for z in range(truck.height - height + 1)
                ),
            )
        else:
            positions = iter([None])

        for position in positions:
            if position == placement.position:
                continue
            candidate = deepcopy(state)
            candidate.placements[package_id].position = position
            if _is_new_valid_neighbor(problem, candidate, seen):
                yield candidate

        rotated_orientations = []
        for axis in ("x", "y", "z"):
            orientation = _rotate_orientation(placement.orientation, axis)
            if orientation not in rotated_orientations:
                rotated_orientations.append(orientation)

        for orientation in rotated_orientations:
            if orientation == placement.orientation:
                continue
            candidate = deepcopy(state)
            candidate.placements[package_id].orientation = orientation
            if _is_new_valid_neighbor(problem, candidate, seen):
                yield candidate

    package_ids = list(state.placements)
    for first_index, first_id in enumerate(package_ids):
        for second_id in package_ids[first_index + 1 :]:
            first_position = state.placements[first_id].position
            second_position = state.placements[second_id].position
            if first_position == second_position:
                continue

            candidate = deepcopy(state)
            candidate.placements[first_id].position = second_position
            candidate.placements[second_id].position = first_position
            if _is_new_valid_neighbor(problem, candidate, seen):
                yield candidate


def _is_new_valid_neighbor(
    problem: Problem,
    candidate: State,
    seen: set[tuple],
) -> bool:
    key = _state_key(candidate)
    if key in seen or not is_valid(problem, candidate):
        return False
    seen.add(key)
    return True


def _state_key(state: State) -> tuple:
    return tuple(
        (
            package_id,
            None
            if placement.position is None
            else (
                placement.position.x,
                placement.position.y,
                placement.position.z,
            ),
            placement.orientation.value,
        )
        for package_id, placement in sorted(state.placements.items())
    )


def _is_inside_truck(problem: Problem, placement: Placement) -> bool:
    if placement.position is None:
        return True

    package = problem.packages[placement.package_id]
    width, length, height = oriented_dimensions(package, placement.orientation)
    x = placement.position.x
    y = placement.position.y
    z = placement.position.z
    truck = problem.truck

    return (
        x >= 0
        and y >= 0
        and z >= 0
        and x + width <= truck.width
        and y + length <= truck.length
        and z + height <= truck.height
    )


def _boxes_overlap(
    problem: Problem,
    first: Placement,
    second: Placement,
) -> bool:
    first_package = problem.packages[first.package_id]
    second_package = problem.packages[second.package_id]
    first_width, first_length, first_height = oriented_dimensions(
        first_package, first.orientation
    )
    second_width, second_length, second_height = oriented_dimensions(
        second_package, second.orientation
    )
    first_position = first.position
    second_position = second.position

    return not (
        first_position.x + first_width <= second_position.x
        or second_position.x + second_width <= first_position.x
        or first_position.y + first_length <= second_position.y
        or second_position.y + second_length <= first_position.y
        or first_position.z + first_height <= second_position.z
        or second_position.z + second_height <= first_position.z
    )


def _has_no_collision(problem: Problem, state: State) -> bool:
    inside_placements = [
        placement
        for placement in state.placements.values()
        if placement.position is not None
    ]

    for index, first in enumerate(inside_placements):
        for second in inside_placements[index + 1 :]:
            if _boxes_overlap(problem, first, second):
                return False

    return True


def _is_within_capacity(problem: Problem, state: State) -> bool:
    return total_weight(problem, state) <= problem.truck.max_capacity


def _overlaps_xy(
    problem: Problem,
    first: Placement,
    second: Placement,
) -> bool:
    first_package = problem.packages[first.package_id]
    second_package = problem.packages[second.package_id]
    first_width, first_length, _ = oriented_dimensions(
        first_package, first.orientation
    )
    second_width, second_length, _ = oriented_dimensions(
        second_package, second.orientation
    )
    first_position = first.position
    second_position = second.position

    return (
        first_position.x < second_position.x + second_width
        and second_position.x < first_position.x + first_width
        and first_position.y < second_position.y + second_length
        and second_position.y < first_position.y + first_length
    )


def _has_support(problem: Problem, state: State, target: Placement) -> bool:
    target_position = target.position
    if target_position.z == 0:
        return True

    has_non_fragile_support = False
    for supporter in state.placements.values():
        if supporter.package_id == target.package_id or supporter.position is None:
            continue

        supporter_package = problem.packages[supporter.package_id]
        _, _, supporter_height = oriented_dimensions(
            supporter_package, supporter.orientation
        )

        if supporter.position.z + supporter_height != target_position.z:
            continue
        if not _overlaps_xy(problem, supporter, target):
            continue

        if supporter_package.is_fragile:
            return False
        has_non_fragile_support = True

    return has_non_fragile_support


def _all_packages_supported(problem: Problem, state: State) -> bool:
    return all(
        _has_support(problem, state, placement)
        for placement in state.placements.values()
        if placement.position is not None
    )


def _move_neighbor(problem: Problem, state: State, rng: random.Random) -> State:
    candidate = deepcopy(state)
    package_id = rng.choice(list(candidate.placements))
    placement = candidate.placements[package_id]

    if rng.random() < 0.25:
        placement.position = None
        return candidate

    package = problem.packages[package_id]
    width, length, height = oriented_dimensions(package, placement.orientation)
    truck = problem.truck

    if width > truck.width or length > truck.length or height > truck.height:
        placement.position = None
        return candidate

    placement.position = Coordinate(
        rng.randint(0, truck.width - width),
        rng.randint(0, truck.length - length),
        rng.randint(0, truck.height - height),
    )
    return candidate


def _rotate_orientation(orientation: Orientation, axis: str) -> Orientation:
    axes = orientation.name

    if axis == "x":
        rotated_axes = axes[0] + axes[2] + axes[1]
    elif axis == "y":
        rotated_axes = axes[2] + axes[1] + axes[0]
    elif axis == "z":
        rotated_axes = axes[1] + axes[0] + axes[2]
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")

    return Orientation[rotated_axes]


def _rotate_neighbor(state: State, rng: random.Random) -> State:
    candidate = deepcopy(state)
    package_id = rng.choice(list(candidate.placements))
    placement = candidate.placements[package_id]
    axis = rng.choice(("x", "y", "z"))
    placement.orientation = _rotate_orientation(placement.orientation, axis)
    return candidate


def _swap_neighbor(state: State, rng: random.Random) -> State:
    candidate = deepcopy(state)
    first_id, second_id = rng.sample(list(candidate.placements), 2)
    first = candidate.placements[first_id]
    second = candidate.placements[second_id]

    first.position, second.position = second.position, first.position
    return candidate

