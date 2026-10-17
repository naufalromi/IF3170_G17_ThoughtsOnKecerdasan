import random
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from src.core import generate_valid_neighbor, is_valid, objective, random_initial_state
from src.models import Problem, State


@dataclass(frozen=True)
class GeneticAlgorithmResult:
    initial_population: tuple[State, ...]
    final_population: tuple[State, ...]
    initial_state: State
    final_state: State
    final_value: float
    max_fitness_history: tuple[float, ...]
    mean_fitness_history: tuple[float, ...]
    population_size: int
    iterations: int
    mutation_rate: float
    duration_seconds: float
    crossover_attempts: int
    invalid_crossovers: int


def genetic_algorithm(
    problem: Problem,
    *,
    population_size: int = 20,
    max_iterations: int = 100,
    mutation_rate: float = 0.1,
    rng: random.Random | None = None,
) -> GeneticAlgorithmResult:
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer")
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("max_iterations must be a non-negative integer")
    if not isinstance(mutation_rate, (int, float)) or not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if any(
        not isfinite(package.value) or package.value < 0
        for package in problem.packages.values()
    ):
        raise ValueError("Package values must be finite and non-negative")
    if not isfinite(sum(package.value for package in problem.packages.values())):
        raise ValueError("Total package value must be finite")

    generator = rng if rng is not None else random.Random()
    started_at = perf_counter()
    package_ids = tuple(sorted(problem.packages))
    population = [random_initial_state(problem, generator) for _ in range(population_size)]
    initial_population = tuple(deepcopy(population))
    fitness = [objective(problem, state) for state in population]
    initial_state = deepcopy(population[max(range(population_size), key=fitness.__getitem__)])
    maxima = [max(fitness)]
    means = [sum(value / population_size for value in fitness)]
    crossover_attempts = 0
    invalid_crossovers = 0

    for _ in range(max_iterations):
        highest = max(fitness)
        weights = [value / highest for value in fitness] if highest > 0 else None
        new_population = []
        for _ in range(population_size):
            first, second = generator.choices(population, weights=weights, k=2)
            child = _crossover(first, second, package_ids, generator)
            if len(package_ids) >= 2:
                crossover_attempts += 1
                if not is_valid(problem, child):
                    invalid_crossovers += 1
                    child = deepcopy(first)

            if generator.random() < mutation_rate:
                mutated = generate_valid_neighbor(problem, child, generator)
                if mutated is not None:
                    child = mutated
            new_population.append(child)

        population = new_population
        fitness = [objective(problem, state) for state in population]
        maxima.append(max(fitness))
        means.append(sum(value / population_size for value in fitness))

    best_index = max(range(population_size), key=fitness.__getitem__)
    return GeneticAlgorithmResult(
        initial_population=initial_population,
        final_population=tuple(deepcopy(population)),
        initial_state=initial_state,
        final_state=deepcopy(population[best_index]),
        final_value=fitness[best_index],
        max_fitness_history=tuple(maxima),
        mean_fitness_history=tuple(means),
        population_size=population_size,
        iterations=max_iterations,
        mutation_rate=mutation_rate,
        duration_seconds=perf_counter() - started_at,
        crossover_attempts=crossover_attempts,
        invalid_crossovers=invalid_crossovers,
    )


def _crossover(
    first: State,
    second: State,
    package_ids: tuple[str, ...],
    rng: random.Random,
) -> State:
    if len(package_ids) < 2:
        return deepcopy(first)
    cut = rng.randrange(1, len(package_ids))
    return State({
        package_id: deepcopy(
            (first if index < cut else second).placements[package_id]
        )
        for index, package_id in enumerate(package_ids)
    })
