import json
import math
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.ga import genetic_algorithm
from src.hill_climbing import (
    random_restart_hill_climbing,
    sideways_hill_climbing,
    steepest_ascent_hill_climbing,
    stochastic_hill_climbing,
)
from src.models import Package, Problem, Truck
from src.output import result_summary, save_result, state_details
from src.simmulated_annealing import simulated_annealing
from src.visualization import draw_history, draw_state

objective_option = {
    "Total value": "value",
    "Value + prioritas ETA": "eta_priority",
}


def get_algorithm_names() -> tuple[str, ...]:
    return (
        "Steepest Ascent Hill-Climbing",
        "Stochastic Hill-Climbing",
        "Hill-Climbing with Sideways Move",
        "Random Restart Hill-Climbing",
        "Simulated Annealing",
        "Genetic Algorithm",
    )


def get_algorithm_code(algorithm: str) -> str:
    names = get_algorithm_names()
    codes = ("steepest", "stochastic", "sideways", "random_restart", "sa", "ga")
    try:
        return codes[names.index(algorithm)]
    except ValueError as exc:
        raise ValueError("Algoritma tidak dikenal.") from exc


def get_algorithm_parameters(algorithm: str) -> tuple[str, ...]:
    configurations = {
        "steepest": (),
        "stochastic": ("max_iterations",),
        "sideways": ("max_sideways_moves",),
        "random_restart": ("max_restarts",),
        "sa": (
            "max_iterations", "initial_temperature", "min_temperature", "cooling_rate"
        ),
        "ga": ("max_iterations", "population_size"),
    }
    return configurations[get_algorithm_code(algorithm)]


def get_parameter_label(name: str) -> str:
    labels = {
        "max_iterations": "Maks. iterasi",
        "max_sideways_moves": "Maks. sideways",
        "max_restarts": "Maks. restart",
        "initial_temperature": "Suhu awal",
        "min_temperature": "Suhu minimum",
        "cooling_rate": "Cooling rate",
        "population_size": "Jumlah populasi",
    }
    try:
        return labels[name]
    except KeyError as exc:
        raise ValueError(f"Parameter {name!r} tidak dikenal.") from exc


def get_parameter_defaults() -> dict[str, str]:
    return {
        "max_iterations": "100",
        "max_sideways_moves": "20",
        "max_restarts": "5",
        "initial_temperature": "10",
        "min_temperature": "0.01",
        "cooling_rate": "0.95",
        "population_size": "20",
    }


def finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} harus berupa angka.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} harus berupa angka berhingga.")
    return number


def integer(value: object, label: str, minimum: int = 0) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} harus berupa bilangan bulat.") from exc
    if number < minimum:
        raise ValueError(f"{label} harus minimal {minimum}.")
    return number


def boolean(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "ya", "yes"}:
            return True
        if normalized in {"false", "tidak", "no"}:
            return False
    raise ValueError(f"{label} harus bernilai true atau false.")


def build_problem(
    truck_data: dict, package_rows: list[dict], objective_mode: str = "value"
) -> Problem:
    truck = Truck(
        integer(truck_data["width"], "Lebar truk", 1),
        integer(truck_data["length"], "Panjang truk", 1),
        integer(truck_data["height"], "Tinggi truk", 1),
        finite_float(truck_data["max_capacity"], "Kapasitas truk"),
    )
    if not package_rows:
        raise ValueError("Tambahkan minimal satu paket.")

    packages = {}
    for index, row in enumerate(package_rows, 1):
        package_id = str(row["id"]).strip()
        if not package_id:
            raise ValueError(f"ID paket baris {index} tidak boleh kosong.")
        if package_id in packages:
            raise ValueError(f"ID paket {package_id!r} dipakai lebih dari sekali.")
        packages[package_id] = Package(
            id=package_id,
            width=integer(row["width"], f"Lebar {package_id}", 1),
            length=integer(row["length"], f"Panjang {package_id}", 1),
            height=integer(row["height"], f"Tinggi {package_id}", 1),
            value=finite_float(row["value"], f"Value {package_id}"),
            weight=finite_float(row["weight"], f"Berat {package_id}"),
            is_fragile=boolean(row["is_fragile"], f"Fragile {package_id}"),
            eta=integer(row["eta"], f"ETA {package_id}"),
        )
    return Problem(truck=truck, packages=packages, objective_mode=objective_mode)


def parse_parameters(algorithm: str, values: dict[str, str]) -> dict:
    parameters = {}
    float_parameters = {
        "initial_temperature", "min_temperature", "cooling_rate"
    }
    for name in get_algorithm_parameters(algorithm):
        label = get_parameter_label(name)
        raw = values[name]
        if name in float_parameters:
            parameters[name] = finite_float(raw, label)
        else:
            minimum = 1 if name == "population_size" else 0
            parameters[name] = integer(raw, label, minimum)
    return parameters


def run_algorithm(problem: Problem, algorithm: str, parameters: dict, seed: int):
    functions = {
        "steepest": steepest_ascent_hill_climbing,
        "stochastic": stochastic_hill_climbing,
        "sideways": sideways_hill_climbing,
        "random_restart": random_restart_hill_climbing,
        "sa": simulated_annealing,
        "ga": genetic_algorithm,
    }
    function = functions[get_algorithm_code(algorithm)]
    return function(problem, rng=random.Random(seed), **parameters)


class TruckSearchApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Thoughts On Kecerdasan - Local Search")
        self.geometry("1320x820")
        self.minsize(1040, 660)
        self.problem: Problem | None = None
        self.result = None
        self.active_algorithm = ""
        self.active_seed = 0

        self.truck_vars = {
            key: tk.StringVar(value=value)
            for key, value in (("width", "5"), ("length", "6"),
                               ("height", "4"), ("max_capacity", "50"))
        }
        self.package_vars = {
            key: tk.StringVar(value=value)
            for key, value in (("id", ""), ("width", "1"), ("length", "1"),
                               ("height", "1"), ("value", "10"),
                               ("weight", "1"), ("eta", "1"))
        }
        self.fragile_var = tk.BooleanVar(value=False)
        self.algorithm_var = tk.StringVar(value=get_algorithm_names()[0])
        self.objective_var = tk.StringVar(value="Total value")
        self.seed_var = tk.StringVar(value="42")
        self.parameter_vars = {
            key: tk.StringVar(value=value)
            for key, value in get_parameter_defaults().items()
        }
        self.status_var = tk.StringVar()

        self._build_interface()
        self._add_examples()
        self._update_parameters()

    def _build_interface(self) -> None:
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        inputs = ttk.Frame(main, padding=8)
        results = ttk.Frame(main, padding=8)
        main.add(inputs, weight=2)
        main.add(results, weight=3)

        truck_frame = ttk.LabelFrame(inputs, text="Truk", padding=8)
        truck_frame.pack(fill=tk.X, pady=(0, 8))
        fields = (("width", "Lebar"), ("length", "Panjang"),
                  ("height", "Tinggi"), ("max_capacity", "Kapasitas"))
        for column, (key, label) in enumerate(fields):
            ttk.Label(truck_frame, text=label).grid(
                row=0, column=column, sticky="w", padx=3
            )
            ttk.Entry(truck_frame, textvariable=self.truck_vars[key], width=11).grid(
                row=1, column=column, sticky="ew", padx=3
            )
            truck_frame.columnconfigure(column, weight=1)

        package_frame = ttk.LabelFrame(inputs, text="Paket", padding=8)
        package_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        columns = ("id", "width", "length", "height", "value", "weight", "fragile", "eta")
        table_frame = ttk.Frame(package_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        self.package_table = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=8
        )
        labels = ("ID", "W", "L", "H", "Value", "Berat", "Fragile", "ETA")
        for column, label in zip(columns, labels):
            self.package_table.heading(column, text=label)
            self.package_table.column(
                column, width=62 if column != "id" else 110, anchor="center"
            )
        table_y_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.package_table.yview
        )
        table_x_scroll = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.package_table.xview
        )
        self.package_table.configure(
            yscrollcommand=table_y_scroll.set, xscrollcommand=table_x_scroll.set
        )
        self.package_table.grid(row=0, column=0, sticky="nsew")
        table_y_scroll.grid(row=0, column=1, sticky="ns")
        table_x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.package_table.bind("<<TreeviewSelect>>", self._select_package)

        editor = ttk.Frame(package_frame)
        editor.pack(fill=tk.X, pady=6)
        editor_fields = (("id", "ID"), ("width", "W"), ("length", "L"),
                         ("height", "H"), ("value", "Value"),
                         ("weight", "Berat"), ("eta", "ETA"))
        for column, (key, label) in enumerate(editor_fields):
            ttk.Label(editor, text=label).grid(
                row=0, column=column, sticky="w", padx=2
            )
            ttk.Entry(editor, textvariable=self.package_vars[key], width=9).grid(
                row=1, column=column, sticky="ew", padx=2
            )
            editor.columnconfigure(column, weight=1)
        ttk.Checkbutton(editor, text="Fragile", variable=self.fragile_var).grid(
            row=2, column=0, sticky="w"
        )

        buttons = ttk.Frame(package_frame)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Tambah / ubah", command=self._save_package).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(buttons, text="Hapus", command=self._delete_package).pack(
            side=tk.LEFT
        )
        ttk.Button(buttons, text="Muat JSON", command=self._load_input).pack(
            side=tk.RIGHT
        )

        settings = ttk.LabelFrame(inputs, text="Pencarian", padding=8)
        settings.pack(fill=tk.X)
        ttk.Label(settings, text="Algoritma").grid(row=0, column=0, sticky="w")
        algorithm_box = ttk.Combobox(
            settings, textvariable=self.algorithm_var, values=get_algorithm_names(),
            state="readonly", width=35,
        )
        algorithm_box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=5)
        algorithm_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._update_parameters()
        )
        ttk.Label(settings, text="Objective").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Combobox(
            settings, textvariable=self.objective_var,
            values=tuple(objective_option), state="readonly", width=35,
        ).grid(row=1, column=1, columnspan=3, sticky="ew", padx=5)
        ttk.Label(settings, text="Seed").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(settings, textvariable=self.seed_var, width=13).grid(
            row=2, column=1, sticky="w", padx=5
        )
        self.parameter_frame = ttk.Frame(settings)
        self.parameter_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=5)
        self.run_button = ttk.Button(
            settings, text="Jalankan", command=self._start_run
        )
        self.run_button.grid(row=4, column=0, columnspan=4, sticky="ew", pady=5)
        settings.columnconfigure(1, weight=1)

        result_tabs = ttk.Notebook(results)
        result_tabs.pack(fill=tk.BOTH, expand=True)
        self.figures = {}
        self.canvases = {}
        for tab_name in ("State awal", "State akhir", "Grafik"):
            frame = ttk.Frame(result_tabs)
            result_tabs.add(frame, text=tab_name)
            figure = Figure(figsize=(6, 5), dpi=100)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figures[tab_name] = figure
            self.canvases[tab_name] = canvas

        details_frame = ttk.Frame(result_tabs)
        result_tabs.add(details_frame, text="Rincian paket")
        self.state_text = tk.Text(details_frame, wrap="none", state="disabled")
        details_y_scroll = ttk.Scrollbar(
            details_frame, orient="vertical", command=self.state_text.yview
        )
        details_x_scroll = ttk.Scrollbar(
            details_frame, orient="horizontal", command=self.state_text.xview
        )
        self.state_text.configure(
            yscrollcommand=details_y_scroll.set, xscrollcommand=details_x_scroll.set
        )
        self.state_text.grid(row=0, column=0, sticky="nsew")
        details_y_scroll.grid(row=0, column=1, sticky="ns")
        details_x_scroll.grid(row=1, column=0, sticky="ew")
        details_frame.rowconfigure(0, weight=1)
        details_frame.columnconfigure(0, weight=1)

        summary_frame = ttk.LabelFrame(results, text="Output", padding=7)
        summary_frame.pack(fill=tk.X, pady=(8, 0))
        self.summary_text = tk.Text(
            summary_frame, height=8, wrap="word", state="disabled"
        )
        self.summary_text.pack(fill=tk.X)
        ttk.Button(
            summary_frame, text="Simpan hasil (JSON + PNG)",
            command=self._save_output,
        ).pack(anchor="e", pady=(5, 0))

        ttk.Label(
            self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w"
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def _add_examples(self) -> None:
        examples = (
            ("P1", 2, 2, 1, 30, 5, "Tidak", 2),
            ("P2", 1, 2, 2, 25, 4, "Tidak", 1),
            ("P3", 1, 1, 1, 15, 2, "Ya", 3),
            ("P4", 2, 1, 1, 20, 3, "Tidak", 2),
        )
        for row in examples:
            self.package_table.insert("", "end", values=row)

    def _select_package(self, _event=None) -> None:
        selected = self.package_table.selection()
        if not selected:
            return
        values = self.package_table.item(selected[0], "values")
        keys = ("id", "width", "length", "height", "value", "weight")
        for key, value in zip(keys, values[:6]):
            self.package_vars[key].set(value)
        self.fragile_var.set(values[6] == "Ya")
        self.package_vars["eta"].set(values[7])

    def _save_package(self) -> None:
        try:
            row = self._editor_row()
            existing = self.package_table.selection()
            rows = self._package_rows()
            if existing:
                selected_id = self.package_table.item(existing[0], "values")[0]
                rows = [item for item in rows if item["id"] != selected_id]
            build_problem(
                {key: var.get() for key, var in self.truck_vars.items()},
                rows + [row],
            )
        except (ValueError, TypeError, KeyError) as exc:
            messagebox.showerror("Input tidak valid", str(exc))
            return
        values = self._row_values(row)
        if existing:
            self.package_table.item(existing[0], values=values)
        else:
            self.package_table.insert("", "end", values=values)
        selected = self.package_table.selection()
        if selected:
            self.package_table.selection_remove(*selected)
        self.package_vars["id"].set("")

    def _delete_package(self) -> None:
        for item_id in self.package_table.selection():
            self.package_table.delete(item_id)

    def _editor_row(self) -> dict:
        row = {key: var.get().strip() for key, var in self.package_vars.items()}
        row["is_fragile"] = self.fragile_var.get()
        return row

    def _package_rows(self) -> list[dict]:
        rows = []
        keys = ("id", "width", "length", "height", "value", "weight", "fragile", "eta")
        for item_id in self.package_table.get_children():
            values = self.package_table.item(item_id, "values")
            row = dict(zip(keys, values))
            row["is_fragile"] = row.pop("fragile") == "Ya"
            rows.append(row)
        return rows

    @staticmethod
    def _row_values(row: dict) -> tuple:
        return (
            row["id"], row["width"], row["length"], row["height"],
            row["value"], row["weight"],
            "Ya" if row["is_fragile"] else "Tidak", row["eta"],
        )

    def _load_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Muat input", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            problem = build_problem(
                data["truck"], data["packages"], data.get("objective_mode", "value")
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("Gagal memuat input", str(exc))
            return
        for key, var in self.truck_vars.items():
            var.set(str(getattr(problem.truck, key)))
        self.objective_var.set(
            next(label for label, mode in objective_option.items()
                 if mode == problem.objective_mode)
        )
        for item_id in self.package_table.get_children():
            self.package_table.delete(item_id)
        for package in problem.packages.values():
            self.package_table.insert(
                "", "end", values=self._row_values(vars(package))
            )
        self.status_var.set(f"Input dimuat dari {Path(path).name}.")

    def _update_parameters(self) -> None:
        for widget in self.parameter_frame.winfo_children():
            widget.destroy()
        for index, name in enumerate(
            get_algorithm_parameters(self.algorithm_var.get())
        ):
            row, column = divmod(index, 2)
            ttk.Label(self.parameter_frame, text=get_parameter_label(name)).grid(
                row=row, column=column * 2, sticky="w", padx=(0, 4), pady=2
            )
            ttk.Entry(
                self.parameter_frame, textvariable=self.parameter_vars[name], width=12
            ).grid(row=row, column=column * 2 + 1, sticky="w", padx=(0, 12), pady=2)

    def _start_run(self) -> None:
        try:
            problem = build_problem(
                {key: var.get() for key, var in self.truck_vars.items()},
                self._package_rows(),
                objective_option[self.objective_var.get()],
            )
            algorithm = self.algorithm_var.get()
            parameters = parse_parameters(
                algorithm, {key: var.get() for key, var in self.parameter_vars.items()}
            )
            seed = integer(self.seed_var.get(), "Seed")
        except (ValueError, TypeError, KeyError) as exc:
            messagebox.showerror("Input tidak valid", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.status_var.set(f"Menjalankan {algorithm}…")
        threading.Thread(
            target=self._run_in_background,
            args=(problem, algorithm, parameters, seed), daemon=True,
        ).start()

    def _run_in_background(
        self, problem: Problem, algorithm: str, parameters: dict, seed: int
    ) -> None:
        try:
            result = run_algorithm(problem, algorithm, parameters, seed)
        except Exception as exc:
            self.after(0, self._show_run_error, str(exc))
        else:
            self.after(0, self._show_result, problem, algorithm, seed, result)

    def _show_run_error(self, error: str) -> None:
        self.run_button.configure(state="normal")
        self.status_var.set("Pencarian gagal.")
        messagebox.showerror("Pencarian gagal", error)

    def _show_result(self, problem: Problem, algorithm: str, seed: int, result) -> None:
        self.problem = problem
        self.result = result
        self.active_algorithm = algorithm
        self.active_seed = seed
        draw_state(self.figures["State awal"], problem, result.initial_state, "State awal")
        draw_state(self.figures["State akhir"], problem, result.final_state, "State akhir")
        draw_history(self.figures["Grafik"], problem, algorithm, result)
        for canvas in self.canvases.values():
            canvas.draw_idle()
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(
            tk.END, result_summary(problem, algorithm, seed, result)
        )
        self.summary_text.configure(state="disabled")
        self.state_text.configure(state="normal")
        self.state_text.delete("1.0", tk.END)
        self.state_text.insert(tk.END, state_details(problem, result))
        self.state_text.configure(state="disabled")
        self.run_button.configure(state="normal")
        self.status_var.set("Selesai. Buka tab untuk melihat state dan grafik.")

    def _save_output(self) -> None:
        if self.result is None or self.problem is None:
            messagebox.showinfo(
                "Belum ada hasil", "Jalankan algoritma terlebih dahulu."
            )
            return
        directory = filedialog.askdirectory(
            title="Pilih folder output",
            initialdir=str(Path(__file__).resolve().parent / "output"),
        )
        if not directory:
            return
        try:
            destination = save_result(
                Path(directory), self.problem, self.active_algorithm,
                self.active_seed, self.result, self.figures,
            )
        except OSError as exc:
            messagebox.showerror("Gagal menyimpan", str(exc))
            return
        self.status_var.set(f"Hasil disimpan di {destination}.")


if __name__ == "__main__":
    TruckSearchApp().mainloop()
