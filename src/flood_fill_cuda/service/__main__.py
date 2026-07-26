"""Run with: uv run python -m flood_fill_cuda.service

Never pass --workers > 1 -- see app.py's _gpu_executor docstring for why
that recreates the exact WSL2 cooperative-launch hazard the single-worker
executor exists to prevent.
"""

import uvicorn


def main():
    uvicorn.run("flood_fill_cuda.service.app:app", host="127.0.0.1",
                port=8000)


if __name__ == "__main__":
    main()
