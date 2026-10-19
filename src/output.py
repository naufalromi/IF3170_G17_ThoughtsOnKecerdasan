import json
from datetime import datetime
from pathlib import Path

from matplotlib.figure import Figure

from src.core import objective, oriented_dimensions, total_weight
from src.models import Problem, State


def state_rows(problem: Problem, state: State) -> list[dict]:
    rows = []
    for package_id in sorted(problem.packages):
        placement = state.placements[package_id]
        package = problem.packages[package_id]
        position = placement.position
        rows.append({
            "id": package_id,
            "position": None if position is None else [position.x, position.y, position.z],
            "orientation": placement.orientation.name,
            "dimensions": list(oriented_dimensions(package, placement.orientation)),
            "value": package.value,
            "weight": package.weight,
            "is_fragile": package.is_fragile,
            "eta": package.eta,
        })
    return rows


def state_details(problem: Problem, result) -> str:
    details = []
    for title, state in (("STATE AWAL", result.initial_state),
                         ("STATE AKHIR", result.final_state)):
        details.append(title)
        for row in state_rows(problem, state):
            position = (
                "di luar truk"
                if row["position"] is None
                else str(tuple(row["position"]))
            )
            details.append(
                f"{row['id']}: {position} | orientasi {row['orientation']} | "
                f"ukuran {tuple(row['dimensions'])} | value {row['value']:g} | "
                f"berat {row['weight']:g} | fragile {row['is_fragile']} | ETA {row['eta']}"
            )
        details.append("")
    return "\n".join(details)


def result_summary(problem: Problem, algorithm: str, seed: int, result) -> str:
    initial_value = objective(problem, result.initial_state)
    final_value = objective(problem, result.final_state)
    loaded = sum(p.position is not None for p in result.final_state.placements.values())
    lines = [
        f"Algoritma: {algorithm}",
        f"Seed: {seed}",
        f"Value awal → akhir: {initial_value:g} → {final_value:g}",
        f"Paket di truk: {loaded}/{len(problem.packages)}",
        f"Berat akhir: {total_weight(problem, result.final_state):g}/{problem.truck.max_capacity:g}",
        f"Iterasi: {result.iterations}",
        f"Durasi: {result.duration_seconds:.4f} detik",
    ]
    if hasattr(result, "termination_reason"):
        lines.append(f"Berhenti karena: {result.termination_reason}")
    variant = getattr(result, "variant", "")
    if variant == "sideways":
        lines.append(f"Sideways move: {result.sideways_moves}")
    elif variant == "random_restart":
        lines.extend((
            f"Restart: {result.restarts}",
            f"Iterasi tiap run: {list(result.iterations_per_restart)}",
        ))
    elif hasattr(result, "acceptance_probabilites"):
        lines.append(f"Stuck: {result.stuck_count}")
    elif hasattr(result, "max_fitness_history"):
        lines.append(f"Populasi: {result.population_size}")
    return "\n".join(lines)


def result_payload(problem: Problem, algorithm: str, seed: int, result) -> dict:
    payload = {
        "algorithm": algorithm,
        "seed": seed,
        "truck": vars(problem.truck),
        "packages": [vars(package) for package in problem.packages.values()],
        "summary": result_summary(problem, algorithm, seed, result),
        "initial_state": state_rows(problem, result.initial_state),
        "final_state": state_rows(problem, result.final_state),
        "final_value": result.final_value,
        "iterations": result.iterations,
        "duration_seconds": result.duration_seconds,
    }
    is_ga = hasattr(result, "max_fitness_history")
    is_sa = hasattr(result, "acceptance_probabilites")
    variant = getattr(result, "variant", "")
    if is_ga:
        payload["max_fitness_history"] = result.max_fitness_history
        payload["mean_fitness_history"] = result.mean_fitness_history
        payload["population_size"] = result.population_size
    else:
        payload["objective_history"] = result.objective_history
    if is_sa:
        payload["acceptance_probabilities"] = result.acceptance_probabilites
        payload["stuck_count"] = result.stuck_count
    if variant == "random_restart":
        payload["restarts"] = result.restarts
        payload["iterations_per_restart"] = result.iterations_per_restart
        payload["restart_boundaries"] = result.restart_boundaries
    if variant == "sideways":
        payload["sideways_moves"] = result.sideways_moves
    return payload


def save_result(
    directory: Path,
    problem: Problem,
    algorithm: str,
    seed: int,
    result,
    figures: dict[str, Figure],
) -> Path:
    stem = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_" + algorithm.lower().replace(" ", "_").replace("-", "_")
    )
    destination = directory / stem
    suffix = 2
    while destination.exists():
        destination = directory / f"{stem}_{suffix}"
        suffix += 1
    destination.mkdir()
    (destination / "result.json").write_text(
        json.dumps(result_payload(problem, algorithm, seed, result),
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for tab_name, filename in (("State awal", "initial_state.png"),
                               ("State akhir", "final_state.png"),
                               ("Grafik", "history.png")):
        figures[tab_name].savefig(destination / filename, dpi=160)
    return destination
