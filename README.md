# CS188 Search & CSP Visualizer

Independent Python/Streamlit educational visualizer for core AI search and CSP algorithms.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

## Set up and run

From the project directory:

```bash
uv sync
uv run streamlit run app.py
```

Then open the local URL printed by Streamlit (usually http://localhost:8501).

To add a dependency:

```bash
uv add package-name
```

To update dependencies and regenerate the lockfile:

```bash
uv lock
uv sync
```

## Included

- Graph search: BFS, DFS, UCS, A*
- Step-by-step frontier, explored set, path-cost display
- CSP: graph coloring (Australia map, a triangulated 3×3 grid that forces backtracking, or a 27-node complex graph) with naive BFS / DFS or backtracking, MRV, LCV, forward checking, arc consistency (AC-3), remaining domains drawn as colored dots in each node, a step-by-step search-tree view, and the frontier / recursion stack / AC-3 arc queue beside it

This is a clean-room educational implementation, not a reproduction of Berkeley's source code or exact interface.
