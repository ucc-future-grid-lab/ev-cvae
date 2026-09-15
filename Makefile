PACKAGE := src/ev_cvae

## Create virtual environment
.venv/bin/activate:
	uv sync --all-groups --all-extras

## Install virtual environment
.PHONY: install
install: .venv/bin/activate

## Update virtual environment
.PHONY: update
update:
	uv sync --upgrade --all-groups --all-extras


## Mypy static checker
.PHONY: mypy
mypy: .venv/bin/activate
	uv run mypy $(PACKAGE)

## Ruff lint
.PHONY: ruff
ruff: .venv/bin/activate
	uv run ruff check $(PACKAGE) 

## Run local CI
.PHONY: local-ci
local-ci: ruff mypy

## Clean files
.PHONY: clean
clean:
	rm -f requirements.txt
	rm -f .coverage
	rm -rf `find . -name __pycache__`
	rm -rf `find . -name .ipynb_checkpoints`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .ipynb_checkpoints
	rm -rf .venv

