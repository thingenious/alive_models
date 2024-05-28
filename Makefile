.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Default target: help"
	@echo ""
	@echo "Targets:"
	@echo " help            Show this message and exit"
	@echo " format          Format the code"
	@echo " lint            Lint the code"
	@echo " forlint         Alias for 'make format && make lint'"
	@echo " clean           Remove unneeded files (__pycache__, .mypy_cache, etc.)"
	@echo " requirements    Generate requirements txt files from pyproject.toml"
	@echo " docs            Generate the documentation"
	@echo " docs-live       Generate the documentation with live reload"
	@echo " build           Build the podman/docker image"
	@echo " start           Start a podman/docker container"
	@echo " start-debug     Start a podman/docker container in debug mode"
	@echo " stop            Stop the podman/docker container"
	@echo " restart         Alias for 'make stop && make start'"
	@echo " dev             Start the python server"
	@echo " compose-up      Start the podman/docker-compose services"
	@echo " compose-down    Stop the podman/docker-compose services"
	@echo " k8s-template    Generate the k8s yaml files using helm"
	@echo " k8s-apply       Apply the generated k8s yaml files"
	@echo " k8s-delete      Delete the k8s deployment"
	@echo ""

.PHONY: format
format:
	isort .
	autoflake --remove-all-unused-imports --remove-unused-variables --in-place .
	black --config pyproject.toml .
	ruff format --config pyproject.toml .

.PHONY: lint
lint:
	isort --check-only .
	black --check --config pyproject.toml .
	mypy --config pyproject.toml .
	flake8 --config=.flake8
	pydocstyle --config pyproject.toml .
	bandit -c pyproject.toml -r .
	yamllint -c .yamllint.yaml .
	ruff check --config pyproject.toml .
	pylint --rcfile=pyproject.toml --recursive y --output-format=text app/

.PHONY: forlint
forlint: format lint

.PHONY: clean
clean:
	python scripts/dev/clean.py

.PHONY: requirements
requirements:
	python -m pip install toml
	python scripts/dev/requirements.py

.PHONY: docs
docs:
	python -m mkdocs build -d site
	@echo "open:   file://`pwd`/site"
	@echo "or use: \`python -m http.server --directory site\`"

.PHONY: docs-live
docs-live:
	python -m mkdocs serve --dev-addr 0.0.0.0:8300 --watch mkdocs.yaml --watch docs --watch app

.PHONY: build
build:
	python scripts/dev/build.py

.PHONY: start
start:
	python scripts/dev/start.py

.PHONY: start-debug
start-debug: build
	python scripts/dev/start.py --debug

.PHONY: stop
stop:
	python scripts/dev/stop.py

.PHONY: restart
restart: stop start

.PHONY: dev
dev:
	python -m app --debug

# deployment
.PHONY: compose-up
compose-up:
	python scripts/dev/deploy.py up

.PHONY: compose-down
compose-down:
	python scripts/dev/deploy.py down

.PHONY: k8s-template
k8s-template:
	python scripts/dev/deploy.py template

.PHONY: k8s-apply
k8s-apply:
	python scripts/dev/deploy.py apply

.PHONY: k8s-delete
k8s-delete:
	python scripts/dev/deploy.py delete
