.PHONY: help setup setup-legacy format lint synthetic biological plots all clean

PYTHON := uv run python

help:
	@echo "Available commands:"
	@echo "  make setup        - Install Python 3 project dependencies via uv"
	@echo "  make setup-legacy - Create isolated Python 2.7 environment for BNFinder2"
	@echo "  make format       - Auto-format code and fix linting errors with Ruff"
	@echo "  make lint         - Check code with Ruff and strict Mypy"
	@echo "  make synthetic    - Run Part I: Synthetic RBN benchmark (N = 5..16)"
	@echo "  make biological   - Run Part II: Mammalian Cell Cycle reconstruction"
	@echo "  make plots        - Generate evaluation figures into assets/"
	@echo "  make all          - Execute full simulation, inference, and plotting pipeline"
	@echo "  make clean        - Remove caches and temporary generated artifacts"

setup:
	uv sync --all-extras

setup-legacy:
	@echo "[*] Setting up legacy Python 2.7 environment for BNFinder2..."
	virtualenv -p python2.7 .venv_legacy
	.venv_legacy/bin/pip install BNFinder

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/

synthetic:
	@echo "[*] Running Part I: Synthetic RBN simulation and DBN inference..."
	$(PYTHON) -m src.run_synthetic

biological:
	@echo "[*] Running Part II: Mammalian Cell Cycle (.bnet) reconstruction..."
	$(PYTHON) -m src.run_biological

plots:
	@echo "[*] Generating publication-grade plots in assets/..."
	$(PYTHON) -m src.plot_results

all: synthetic biological plots

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	rm -rf data/networks/*.json data/datasets/*.txt
	find data/results -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +