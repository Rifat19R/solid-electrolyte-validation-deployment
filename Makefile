.PHONY: setup test lint format typecheck preflight check freeze-env

setup:
	./setup_wsl.sh

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src

preflight:
	python scripts/preflight.py

check: preflight test lint typecheck
	ruff format --check .

freeze-env:
	python -m pip freeze | sort > environment/pip-freeze.txt
