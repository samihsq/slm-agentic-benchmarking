"""
Matrix Recall Task Generator

Generates random integer matrices and natural-language lookup questions
that isolate *recall* — the ability to locate and return specific values
from structured tabular data without any computation.

Difficulty progression:
  easy   – single cell, explicit (row, col) on 10×10
  medium – ordinal / named positions, full row/column on 10×10
  hard   – multi-cell lookups, sub-matrix, diagonal on 10×10
  x-hard – same question styles but on a 15×15 matrix with larger
           extractions (5–8 cells, 4×4+ sub-matrices, scattered reads)
"""

import random
from typing import Any, Dict, List, Tuple

ORDINALS = [
    "", "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
]

MATRIX_SIZE = {"easy": 10, "medium": 10, "hard": 10, "x-hard": 15}


def _corners(size: int) -> Dict[str, Tuple[int, int]]:
    return {
        "top-left corner":     (1, 1),
        "top-right corner":    (1, size),
        "bottom-left corner":  (size, 1),
        "bottom-right corner": (size, size),
    }


def _centers(size: int) -> Dict[str, Tuple[int, int]]:
    mid = (size + 1) // 2
    return {
        "center":        (mid, mid),
        "center-left":   (mid, 1),
        "center-right":  (mid, size),
        "top-center":    (1, mid),
        "bottom-center": (size, mid),
    }


def gen_matrix(size: int = 10, low: int = 10, high: int = 99) -> List[List[int]]:
    """Generate a *size* × *size* matrix of random two-digit integers."""
    return [[random.randint(low, high) for _ in range(size)] for _ in range(size)]


def format_matrix(matrix: List[List[int]]) -> str:
    """Render a matrix as a labelled text grid for the prompt."""
    size = len(matrix)
    cw = 7
    hdr = " " * 6 + "".join(f"{'Col' + str(c + 1):>{cw}}" for c in range(size))
    lines = [hdr]
    for r in range(size):
        label = f"{'Row' + str(r + 1):<6}"
        cells = "".join(f"{matrix[r][c]:>{cw}d}" for c in range(size))
        lines.append(label + cells)
    return "\n".join(lines)


# ── Question factories (size-aware) ────────────────────────────────

def _q_single_cell_explicit(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    r, c = rng.randint(1, sz), rng.randint(1, sz)
    templates = [
        f"What is the value at row {r}, column {c}?",
        f"What is the element in position ({r}, {c})?",
        f"Return the value located at row {r}, column {c}.",
        f"Look up row {r}, column {c} in the matrix.",
    ]
    return {"question": rng.choice(templates), "answer": matrix[r-1][c-1],
            "answer_type": "single", "coordinates": [(r, c)]}


def _q_single_cell_ordinal(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    r, c = rng.randint(1, min(sz, len(ORDINALS)-1)), rng.randint(1, min(sz, len(ORDINALS)-1))
    templates = [
        f"What is the value in the {ORDINALS[r]} row, {ORDINALS[c]} column?",
        f"Find the element at the {ORDINALS[r]} row and {ORDINALS[c]} column.",
        f"Return the number in the {ORDINALS[r]} row and the {ORDINALS[c]} column of the matrix.",
    ]
    return {"question": rng.choice(templates), "answer": matrix[r-1][c-1],
            "answer_type": "single", "coordinates": [(r, c)]}


def _q_named_position(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    all_positions = {**_corners(sz), **_centers(sz)}
    name = rng.choice(list(all_positions))
    r, c = all_positions[name]
    templates = [
        f"What value is at the {name} of the matrix?",
        f"Return the element at the {name}.",
        f"What number is located at the {name}?",
    ]
    return {"question": rng.choice(templates), "answer": matrix[r-1][c-1],
            "answer_type": "single", "coordinates": [(r, c)]}


def _q_full_row(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    r = rng.randint(1, sz)
    templates = [
        f"List all values in row {r}.",
        f"What are the values in row {r}, from left to right?",
        f"Return every element in row {r} as a list.",
    ]
    return {"question": rng.choice(templates), "answer": list(matrix[r-1]),
            "answer_type": "list", "coordinates": [(r, c) for c in range(1, sz+1)]}


def _q_full_column(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    c = rng.randint(1, sz)
    templates = [
        f"List all values in column {c}.",
        f"What are the values in column {c}, from top to bottom?",
        f"Return every element in column {c} as a list.",
    ]
    return {"question": rng.choice(templates),
            "answer": [matrix[r][c-1] for r in range(sz)],
            "answer_type": "list", "coordinates": [(r, c) for r in range(1, sz+1)]}


def _q_multi_cell(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    n = rng.randint(2, 4)
    cells: List[Tuple[int, int]] = []
    while len(cells) < n:
        cell = (rng.randint(1, sz), rng.randint(1, sz))
        if cell not in cells:
            cells.append(cell)
    coords_str = ", ".join(f"({r}, {c})" for r, c in cells)
    answer = [matrix[r-1][c-1] for r, c in cells]
    templates = [
        f"What are the values at positions {coords_str}?",
        f"Return the values located at {coords_str} as a list.",
        f"Look up the values at {coords_str} in the matrix.",
    ]
    return {"question": rng.choice(templates), "answer": answer,
            "answer_type": "list", "coordinates": cells}


def _q_submatrix(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    sub_size = rng.choice([2, 3])
    r_start = rng.randint(1, sz - sub_size + 1)
    c_start = rng.randint(1, sz - sub_size + 1)
    sub = [[matrix[r_start-1+dr][c_start-1+dc] for dc in range(sub_size)]
           for dr in range(sub_size)]
    templates = [
        f"Return the {sub_size}x{sub_size} sub-matrix starting at row {r_start}, column {c_start}.",
        f"Extract the {sub_size}x{sub_size} block whose top-left cell is row {r_start}, column {c_start}.",
    ]
    return {"question": rng.choice(templates), "answer": sub,
            "answer_type": "matrix",
            "coordinates": [(r_start+dr, c_start+dc)
                            for dr in range(sub_size) for dc in range(sub_size)]}


def _q_diagonal(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    diag_type = rng.choice(["main", "anti"])
    if diag_type == "main":
        answer = [matrix[i][i] for i in range(sz)]
        question = rng.choice([
            f"What are the values on the main diagonal (top-left to bottom-right)?",
            f"List the diagonal values from row 1 col 1 to row {sz} col {sz}.",
        ])
        coords = [(i+1, i+1) for i in range(sz)]
    else:
        answer = [matrix[i][sz-1-i] for i in range(sz)]
        question = rng.choice([
            f"What are the values on the anti-diagonal (top-right to bottom-left)?",
            f"List the diagonal values from row 1 col {sz} to row {sz} col 1.",
        ])
        coords = [(i+1, sz-i) for i in range(sz)]
    return {"question": question, "answer": answer,
            "answer_type": "list", "coordinates": coords}


def _q_row_slice(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    sz = len(matrix)
    r = rng.randint(1, sz)
    c_start = rng.randint(1, sz - 2)
    c_end = rng.randint(c_start + 2, sz)
    answer = [matrix[r-1][c-1] for c in range(c_start, c_end+1)]
    templates = [
        f"List the values in row {r} from column {c_start} to column {c_end}.",
        f"Return elements in row {r}, columns {c_start} through {c_end}.",
    ]
    return {"question": rng.choice(templates), "answer": answer,
            "answer_type": "list",
            "coordinates": [(r, c) for c in range(c_start, c_end+1)]}


# ── x-hard factories (15×15 specific) ──────────────────────────────

def _q_xh_many_cells(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    """5-8 scattered cell lookups on a large matrix."""
    sz = len(matrix)
    n = rng.randint(5, 8)
    cells: List[Tuple[int, int]] = []
    while len(cells) < n:
        cell = (rng.randint(1, sz), rng.randint(1, sz))
        if cell not in cells:
            cells.append(cell)
    coords_str = ", ".join(f"({r}, {c})" for r, c in cells)
    answer = [matrix[r-1][c-1] for r, c in cells]
    templates = [
        f"What are the values at positions {coords_str}?",
        f"Return the values located at {coords_str} as a list.",
    ]
    return {"question": rng.choice(templates), "answer": answer,
            "answer_type": "list", "coordinates": cells}


def _q_xh_large_submatrix(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    """4×4 or 5×5 sub-matrix extraction."""
    sz = len(matrix)
    sub_size = rng.choice([4, 5])
    r_start = rng.randint(1, sz - sub_size + 1)
    c_start = rng.randint(1, sz - sub_size + 1)
    sub = [[matrix[r_start-1+dr][c_start-1+dc] for dc in range(sub_size)]
           for dr in range(sub_size)]
    templates = [
        f"Return the {sub_size}x{sub_size} sub-matrix starting at row {r_start}, column {c_start}.",
        f"Extract the {sub_size}x{sub_size} block whose top-left cell is row {r_start}, column {c_start}.",
    ]
    return {"question": rng.choice(templates), "answer": sub,
            "answer_type": "matrix",
            "coordinates": [(r_start+dr, c_start+dc)
                            for dr in range(sub_size) for dc in range(sub_size)]}


def _q_xh_two_rows(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    """Extract two full rows."""
    sz = len(matrix)
    r1, r2 = sorted(rng.sample(range(1, sz+1), 2))
    answer = list(matrix[r1-1]) + list(matrix[r2-1])
    templates = [
        f"List all values in row {r1} followed by all values in row {r2}.",
        f"Return the elements of row {r1} and row {r2} concatenated into one list.",
    ]
    return {"question": rng.choice(templates), "answer": answer,
            "answer_type": "list",
            "coordinates": [(r1, c) for c in range(1, sz+1)]
                         + [(r2, c) for c in range(1, sz+1)]}


def _q_xh_row_and_column(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    """Extract a full row and a full column (concatenated)."""
    sz = len(matrix)
    r = rng.randint(1, sz)
    c = rng.randint(1, sz)
    row_vals = list(matrix[r-1])
    col_vals = [matrix[ri][c-1] for ri in range(sz)]
    answer = row_vals + col_vals
    templates = [
        f"List all values in row {r}, then list all values in column {c}. Combine both into one list.",
        f"Return the elements of row {r} followed by the elements of column {c} as a single list.",
    ]
    return {"question": rng.choice(templates), "answer": answer,
            "answer_type": "list",
            "coordinates": [(r, ci) for ci in range(1, sz+1)]
                         + [(ri, c) for ri in range(1, sz+1)]}


def _q_xh_scattered_slices(matrix: List[List[int]], rng: random.Random) -> Dict[str, Any]:
    """Multiple non-contiguous row slices."""
    sz = len(matrix)
    n_slices = rng.randint(2, 3)
    answer: List[int] = []
    coords: List[Tuple[int, int]] = []
    parts: List[str] = []
    for _ in range(n_slices):
        r = rng.randint(1, sz)
        c_start = rng.randint(1, sz - 2)
        c_end = rng.randint(c_start + 1, min(c_start + 4, sz))
        for c in range(c_start, c_end + 1):
            answer.append(matrix[r-1][c-1])
            coords.append((r, c))
        parts.append(f"row {r} columns {c_start}-{c_end}")
    desc = ", then ".join(parts)
    return {"question": f"List the values from {desc}, concatenated into one list.",
            "answer": answer, "answer_type": "list", "coordinates": coords}


# ── Difficulty → factory mapping ────────────────────────────────────

EASY_FACTORIES = [_q_single_cell_explicit]
MEDIUM_FACTORIES = [_q_single_cell_ordinal, _q_named_position, _q_full_row, _q_full_column]
HARD_FACTORIES = [_q_multi_cell, _q_submatrix, _q_diagonal, _q_row_slice]
XHARD_FACTORIES = [
    _q_xh_many_cells, _q_xh_large_submatrix, _q_xh_two_rows,
    _q_xh_row_and_column, _q_xh_scattered_slices,
    _q_diagonal, _q_multi_cell,  # reuse on 15×15
]

DIFFICULTY_FACTORIES = {
    "easy": EASY_FACTORIES,
    "medium": MEDIUM_FACTORIES,
    "hard": HARD_FACTORIES,
    "x-hard": XHARD_FACTORIES,
}


class MatrixRecallTaskGenerator:
    """Generate matrix recall tasks at configurable difficulty."""

    def __init__(self, seed: int = 42, matrix_size: int = 10, value_range: Tuple[int, int] = (10, 99)):
        self.seed = seed
        self.matrix_size = matrix_size
        self.value_low, self.value_high = value_range

    def generate(
        self,
        num_tasks: int = 200,
        difficulty_distribution: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        if difficulty_distribution is None:
            difficulty_distribution = {
                "easy": 0.30, "medium": 0.25, "hard": 0.25, "x-hard": 0.20,
            }

        rng = random.Random(self.seed)

        difficulties: List[str] = []
        for diff, prop in difficulty_distribution.items():
            difficulties.extend([diff] * int(num_tasks * prop))
        while len(difficulties) < num_tasks:
            difficulties.append(rng.choice(list(difficulty_distribution)))
        rng.shuffle(difficulties)

        tasks: List[Dict[str, Any]] = []
        for idx, diff in enumerate(difficulties[:num_tasks]):
            size = MATRIX_SIZE.get(diff, self.matrix_size)
            matrix = gen_matrix(size, self.value_low, self.value_high)
            factory = rng.choice(DIFFICULTY_FACTORIES[diff])
            q_data = factory(matrix, rng)

            tasks.append({
                "task_id": f"matrix_recall_{idx:04d}",
                "difficulty": diff,
                "matrix_size": size,
                "matrix": matrix,
                "matrix_text": format_matrix(matrix),
                "question": q_data["question"],
                "answer": q_data["answer"],
                "answer_type": q_data["answer_type"],
                "coordinates": q_data["coordinates"],
            })

        return tasks
