from matplotlib.figure import Figure

from src.core import objective, oriented_dimensions
from src.models import Problem, State


def _box_faces(x: int, y: int, z: int, w: int, length: int, h: int):
    a = (x, y, z)
    b = (x + w, y, z)
    c = (x + w, y + length, z)
    d = (x, y + length, z)
    e = (x, y, z + h)
    f = (x + w, y, z + h)
    g = (x + w, y + length, z + h)
    i = (x, y + length, z + h)
    return ([a, b, c, d], [e, f, g, i], [a, b, f, e],
            [b, c, g, f], [c, d, i, g], [d, a, e, i])


def draw_state(figure: Figure, problem: Problem, state: State, title: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure.clear()
    ax = figure.add_subplot(111, projection="3d")
    truck = problem.truck
    edges = (
        ((0, 0, 0), (truck.width, 0, 0)),
        ((0, truck.length, 0), (truck.width, truck.length, 0)),
        ((0, 0, truck.height), (truck.width, 0, truck.height)),
        ((0, truck.length, truck.height), (truck.width, truck.length, truck.height)),
        ((0, 0, 0), (0, truck.length, 0)),
        ((truck.width, 0, 0), (truck.width, truck.length, 0)),
        ((0, 0, truck.height), (0, truck.length, truck.height)),
        ((truck.width, 0, truck.height), (truck.width, truck.length, truck.height)),
        ((0, 0, 0), (0, 0, truck.height)),
        ((truck.width, 0, 0), (truck.width, 0, truck.height)),
        ((0, truck.length, 0), (0, truck.length, truck.height)),
        ((truck.width, truck.length, 0), (truck.width, truck.length, truck.height)),
    )
    for start, end in edges:
        ax.plot(*zip(start, end), color="#64748b", linewidth=1)

    colors = ("#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2")
    outside = []
    for index, package_id in enumerate(sorted(problem.packages)):
        placement = state.placements[package_id]
        if placement.position is None:
            outside.append(package_id)
            continue
        position = placement.position
        w, length, h = oriented_dimensions(
            problem.packages[package_id], placement.orientation
        )
        box = Poly3DCollection(
            _box_faces(position.x, position.y, position.z, w, length, h),
            facecolors=colors[index % len(colors)],
            edgecolors="#1e293b", linewidths=0.7, alpha=0.63,
        )
        ax.add_collection3d(box)
        ax.text(position.x + w / 2, position.y + length / 2,
                position.z + h / 2, package_id, color="black",
                ha="center", va="center", fontsize=8)

    ax.set(xlim=(0, truck.width), ylim=(0, truck.length), zlim=(0, truck.height),
           xlabel="X / lebar", ylabel="Y / panjang", zlabel="Z / tinggi")
    ax.set_box_aspect((truck.width, truck.length, truck.height))
    ax.set_title(f"{title} · value {objective(problem, state):g}")
    figure.text(
        0.02, 0.02,
        "Di luar truk: " + (", ".join(outside) if outside else "tidak ada"),
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))


def draw_history(figure: Figure, algorithm: str, result) -> None:
    figure.clear()
    if hasattr(result, "max_fitness_history"):
        ax = figure.add_subplot(111)
        ax.plot(result.max_fitness_history, label="Maksimum", color="#2563eb")
        ax.plot(result.mean_fitness_history, label="Rata-rata", color="#d97706")
        ax.legend()
    elif hasattr(result, "acceptance_probabilites"):
        ax = figure.add_subplot(211)
        ax.plot(result.objective_history, color="#2563eb")
        probability_ax = figure.add_subplot(212)
        probability_ax.plot(
            range(1, len(result.acceptance_probabilites) + 1),
            result.acceptance_probabilites, color="#d97706",
        )
        probability_ax.set(xlabel="Iterasi", ylabel="Probabilitas", ylim=(0, 1.05))
    else:
        ax = figure.add_subplot(111)
        ax.plot(result.objective_history, color="#2563eb")
        if getattr(result, "variant", "") == "random_restart":
            for boundary in result.restart_boundaries[1:]:
                ax.axvline(boundary, color="#d97706", linestyle="--", linewidth=1)
    ax.set(xlabel="Iterasi", ylabel="Value", title="Perkembangan objective")
    ax.grid(alpha=0.3)
    figure.tight_layout()
