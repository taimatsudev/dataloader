install:
	poetry install
lint:
	poetry run pylint -d duplicate-code **/*.py
test: install
	poetry run python -m unittest tests/unit/test_*.py
