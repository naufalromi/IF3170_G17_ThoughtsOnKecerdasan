from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from math import isclose
from time import perf_counter

from src.core import (
    generate_valid_neighbor,
    generate_valid_neighbors,
    is_valid,
    objective,
    random_initial_state,
)
from src.models import Problem, State


@dataclass(frozen=True)
class HillClimbingResult:
    variant: str
    initial_state: State
    final_state: State
    final_value: float
    objective_history: tuple[float, ...]
    iterations: int
    duration_seconds: float
    termination_reason: str
    sideways_moves: int = 0
    restarts: int = 0
    iterations_per_restart: tuple[int, ...] = ()
    restart_boundaries: tuple[int, ...] = ()


@dataclass(frozen=True)
class _RunResult:
    initial_state: State
    final_state: State
    history: tuple[float, ...]
    iterations: int
    termination_reason: str
    sideways_moves: int = 0


def steepest_ascent_hill_climbing(
    problem: Problem,
    initial_state: State | None = None,
    *,
    max_iterations: int = 1_000,
    rng: random.Random | None = None,
) -> HillClimbingResult:
    _validate_maximum("max_iterations", max_iterations)
    generator = rng or random.Random()
    start_state = _prepare_initial_state(problem, initial_state, generator)

    started_at = perf_counter()
    run = _steepest_run(problem, start_state, max_iterations, generator)
    duration = perf_counter() - started_at

    return _to_result("steepest_ascent", run, duration)


def stochastic_hill_climbing(
    problem: Problem,
    initial_state: State | None = None,
    *,
    max_iterations: int = 1_000,
    rng: random.Random | None = None,
) -> HillClimbingResult:
    _validate_maximum("max_iterations", max_iterations)
    generator = rng or random.Random()
    start_state = _prepare_initial_state(problem, initial_state, generator)

    started_at = perf_counter()
    current = deepcopy(start_state)
    current_value = objective(problem, current)
    history = [current_value]
    iterations = 0
    termination_reason = "max_iterations"

    while iterations < max_iterations:
        neighbor = generate_valid_neighbor(problem, current, generator)
        if neighbor is not None:
            neighbor_value = objective(problem, neighbor)
            if _is_greater(neighbor_value, current_value):
                current = neighbor
                current_value = neighbor_value

        history.append(current_value)
        iterations += 1

    duration = perf_counter() - started_at
    run = _RunResult(
        deepcopy(start_state),
        deepcopy(current),
        tuple(history),
        iterations,
        termination_reason,
    )
    return _to_result("stochastic", run, duration)


def sideways_hill_climbing(
    problem: Problem,
    initial_state: State | None = None,
    *,
    max_sideways_moves: int,
    max_iterations: int = 1_000,
    rng: random.Random | None = None,
) -> HillClimbingResult:
    _validate_maximum("max_sideways_moves", max_sideways_moves)
    _validate_maximum("max_iterations", max_iterations)
    generator = rng or random.Random()
    start_state = _prepare_initial_state(problem, initial_state, generator)

    started_at = perf_counter()
    current = deepcopy(start_state)
    current_value = objective(problem, current)
    history = [current_value]
    iterations = 0
    consecutive_sideways = 0
    total_sideways = 0
    termination_reason = "max_iterations"

    while iterations < max_iterations:
        neighbor, neighbor_value = _best_neighbor(
            problem, current, current_value, generator, allow_equal=True
        )
        if neighbor is None:
            termination_reason = "local_optimum"
            break

        if _is_equal(neighbor_value, current_value):
            if consecutive_sideways >= max_sideways_moves:
                termination_reason = "max_sideways_moves"
                break
            consecutive_sideways += 1
            total_sideways += 1
        else:
            consecutive_sideways = 0

        current = neighbor
        current_value = neighbor_value
        history.append(current_value)
        iterations += 1

    duration = perf_counter() - started_at
    run = _RunResult(
        deepcopy(start_state),
        deepcopy(current),
        tuple(history),
        iterations,
        termination_reason,
        total_sideways,
    )
    return _to_result("sideways", run, duration)


def random_restart_hill_climbing(
    problem: Problem,
    initial_state: State | None = None,
    *,
    max_restarts: int,
    max_iterations_per_restart: int = 1_000,
    rng: random.Random | None = None,
) -> HillClimbingResult:
    _validate_maximum("max_restarts", max_restarts)
    _validate_maximum("max_iterations_per_restart", max_iterations_per_restart)
    generator = rng or random.Random()
    first_state = _prepare_initial_state(problem, initial_state, generator)

    started_at = perf_counter()
    runs: list[_RunResult] = []
    best_state: State | None = None
    best_value = float("-inf")
    optimum_upper_bound = sum(package.value for package in problem.packages.values())

    for run_index in range(max_restarts + 1):
        run_state = (
            first_state
            if run_index == 0
            else random_initial_state(problem, generator)
        )
        run = _steepest_run(
            problem, run_state, max_iterations_per_restart, generator
        )
        runs.append(run)
        run_value = objective(problem, run.final_state)

        if best_state is None or _is_greater(run_value, best_value):
            best_state = deepcopy(run.final_state)
            best_value = run_value

        if _is_equal(best_value, optimum_upper_bound):
            break

    duration = perf_counter() - started_at
    assert best_state is not None

    history: list[float] = []
    boundaries: list[int] = []
    for run in runs:
        boundaries.append(len(history))
        history.extend(run.history)

    reached_upper_bound = _is_equal(best_value, optimum_upper_bound)
    return HillClimbingResult(
        variant="random_restart",
        initial_state=deepcopy(first_state),
        final_state=best_state,
        final_value=best_value,
        objective_history=tuple(history),
        iterations=sum(run.iterations for run in runs),
        duration_seconds=duration,
        termination_reason=(
            "global_upper_bound" if reached_upper_bound else "max_restarts"
        ),
        restarts=len(runs) - 1,
        iterations_per_restart=tuple(run.iterations for run in runs),
        restart_boundaries=tuple(boundaries),
    )


def _steepest_run(
    problem: Problem,
    initial_state: State,
    max_iterations: int,
    rng: random.Random,
) -> _RunResult:
    current = deepcopy(initial_state)
    current_value = objective(problem, current)
    history = [current_value]
    iterations = 0
    termination_reason = "max_iterations"

    while iterations < max_iterations:
        neighbor, neighbor_value = _best_neighbor(
            problem, current, current_value, rng, allow_equal=False
        )
        if neighbor is None:
            termination_reason = "local_optimum"
            break

        current = neighbor
        current_value = neighbor_value
        history.append(current_value)
        iterations += 1

    return _RunResult(
        deepcopy(initial_state),
        deepcopy(current),
        tuple(history),
        iterations,
        termination_reason,
    )


def _best_neighbor(
    problem: Problem,
    state: State,
    current_value: float,
    rng: random.Random,
    *,
    allow_equal: bool,
) -> tuple[State | None, float]:
    best_value = float("-inf")
    best_neighbor = None
    equal_best_count = 0

    for neighbor in generate_valid_neighbors(problem, state):
        neighbor_value = objective(problem, neighbor)
        acceptable = _is_greater(neighbor_value, current_value) or (
            allow_equal and _is_equal(neighbor_value, current_value)
        )
        if not acceptable:
            continue

        if _is_greater(neighbor_value, best_value):
            best_value = neighbor_value
            best_neighbor = neighbor
            equal_best_count = 1
        elif _is_equal(neighbor_value, best_value):
            equal_best_count += 1
            if rng.randrange(equal_best_count) == 0:
                best_neighbor = neighbor

    if best_neighbor is None:
        return None, current_value
    return best_neighbor, best_value


def _prepare_initial_state(
    problem: Problem,
    initial_state: State | None,
    rng: random.Random,
) -> State:
    state = random_initial_state(problem, rng) if initial_state is None else initial_state
    if not is_valid(problem, state):
        raise ValueError("initial_state must satisfy all problem constraints")
    return deepcopy(state)


def _to_result(
    variant: str,
    run: _RunResult,
    duration: float,
) -> HillClimbingResult:
    return HillClimbingResult(
        variant=variant,
        initial_state=run.initial_state,
        final_state=run.final_state,
        final_value=run.history[-1],
        objective_history=run.history,
        iterations=run.iterations,
        duration_seconds=duration,
        termination_reason=run.termination_reason,
        sideways_moves=run.sideways_moves,
    )


def _validate_maximum(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _is_equal(first: float, second: float) -> bool:
    return isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def _is_greater(first: float, second: float) -> bool:
    return first > second and not _is_equal(first, second)
