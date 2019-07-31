init:
	pip install -r requirements.txt

check:
	pyflakes .
	mypy .

format:
	black .
	autoflake --in-place --recursive --remove-all-unused-imports --ignore-init-module-imports .
	isort --recursive .

test:
	py.test tests

.PHONY: init check format test
