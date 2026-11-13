.PHONY: fmt lint test check clean setup-dev-env upload-pypi upload-anaconda
#.PHONY: fmt lint type test check clean setup-dev-env upload-pypi upload-anaconda

fmt:
	ruff format

lint:
	ruff check --fix

#type:
#	python -m mypy

test:
	pytest

check: fmt lint

setup-dev-env:
	bash scripts/setup-dev-env.sh

upload-pypi:
	bash scripts/upload2pypi.sh

upload-anaconda:
	bash scripts/upload2anaconda.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
