from __future__ import annotations

import random
import math
from copy import deepcopy
from dataclasses import dataclass
from math import isclose
from time import perf_counter

from src.core import generate_valid_neighbor, is_valid, objective, random_initial_state
from src.hill_climbing import _prepare_initial_state
from src.models import Problem, State


@dataclass (frozen=True)
class SimulatedAnnealingResults:
    initial_state: State
    final_state: State
    final_value: float
    objective_history: tuple[float, ...]
    iterations: int
    duration_seconds: float
    termination_reason: str
    acceptance_probabilites: tuple[float, ...] = ()
    stuck_count: int = 0

@dataclass(frozen=True)
class _RunResultAnnealing:
    initial_state: State
    final_state: State
    history: tuple[float, ...]
    iterations: int
    termination_reason: str
    acceptance_probabilities: tuple[float, ...] = ()
    stuck_count: int = 0

def _to_result(
    variant: str,
    run: _RunResultAnnealing,
    duration: float,
) -> SimulatedAnnealingResults:
    return SimulatedAnnealingResults(
        initial_state=run.initial_state,
        final_state=run.final_state,
        final_value=run.history[-1],
        objective_history=run.history,
        iterations=run.iterations,
        duration_seconds=duration,
        termination_reason=run.termination_reason,
        acceptance_probabilites=run.acceptance_probabilities,
        stuck_count=run.stuck_count
    )


def simulated_annealing(
    problem: Problem,
    initial_state: State | None = None,
    *,
    initial_temperature: float | None = None,
    min_temperature: float = 1e-3,
    cooling_rate: float = 0.995,
    max_iterations: int = 100_000,
    rng: random.Random | None = None,
) -> SimulatedAnnealingResults:
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1 (exclusive)")
    if min_temperature <= 0:
        raise ValueError("min_temperature must be positive")

    generator = rng or random.Random()
    start_state = _prepare_initial_state(problem, initial_state, generator)

    if initial_temperature is None:
        initial_temperature = _estimate_initial_temperature(
            problem, start_state, generator
        )
    if initial_temperature <= min_temperature:
        raise ValueError("initial_temperature must be greater than min_temperature")

    started_at = perf_counter()
    current = deepcopy(start_state)
    current_value = objective(problem, current)
    best, best_value = deepcopy(current), current_value

    history = [current_value]
    acceptance_history: list[float] = []
    stuck_count = 0
    no_neighbor_count = 0

    temperature = initial_temperature
    iterations = 0
    termination_reason = "temperature_min"

    while temperature > min_temperature:
        if iterations >= max_iterations:
            termination_reason = "max_iterations"
            break
        iterations += 1

        neighbor = generate_valid_neighbor(problem, current, generator)
        probability = 1.0

        if neighbor is None:
            no_neighbor_count += 1
        else:
            neighbor_value = objective(problem, neighbor)
            delta = neighbor_value - current_value

            if delta >= 0:
                accepted = True
            else:
                probability = math.exp(delta / temperature)
                accepted = generator.random() < probability
                if not accepted:
                    stuck_count += 1  # worsening move rejected -> stayed put

            if accepted:
                current, current_value = neighbor, neighbor_value
                if current_value > best_value:
                    best, best_value = deepcopy(current), current_value

        acceptance_history.append(probability)
        history.append(current_value)
        temperature *= cooling_rate

    duration = perf_counter() - started_at
    run = _RunResultAnnealing(
        deepcopy(start_state),
        best,
        tuple(history),
        iterations,
        termination_reason,
        acceptance_probabilities=tuple(acceptance_history),
        stuck_count=stuck_count,
    )
    return _to_result("simulated_annealing", run, duration)


def _estimate_initial_temperature(problem, state, rng, samples=50, accept_prob=0.8):
    base = objective(problem, state)
    drops = []
    for _ in range(samples):
        neighbor = generate_valid_neighbor(problem, state, rng)
        if neighbor is None:
            continue
        delta = objective(problem, neighbor) - base
        if delta < 0:
            drops.append(-delta)
    if not drops:
        return 1.0
    return (sum(drops) / len(drops)) / -math.log(accept_prob)
