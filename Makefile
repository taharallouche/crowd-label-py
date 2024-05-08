.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 lint/black
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache


typing-py: ## check typing
	mypy size_matters --follow-imports=skip --ignore-missing-imports

lint:
	ruff check

test:  ## run tests quickly with the default Python
	pytest -vvv

test-all: lint test

coverage: ## check code coverage quickly with the default Python
	coverage run --source size_matters -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

build-docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/size_matters.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ size_matters
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

docs: build-docs ## Generate and show the Sphinx HTML documentation
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload --config-file ../.pypirc dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

install-dev: clean ## install the package to the active Python's site-packages
	pip install -r requirements.txt
	python setup.py develop

test-release: dist ## test package and upload a release
	twine upload --repository testpypi --config-file ../.pypirc dist/*
