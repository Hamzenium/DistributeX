# ===============================
# Project Configuration
# ===============================
APP=app.main:app
HOST=0.0.0.0
PORT=8000

VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

COMPOSE=docker compose

.DEFAULT_GOAL := help

# ===============================
# Help
# ===============================
help:
	@echo ""
	@echo "Local development:"
	@echo "  make venv        Create virtual environment"
	@echo "  make install     Install dependencies"
	@echo "  make run         Run FastAPI locally"
	@echo ""
	@echo "Docker:"
	@echo "  make build       Build Docker images"
	@echo "  make up          Start services with Docker"
	@echo "  make down        Stop Docker services"
	@echo ""
	@echo "Utilities:"
	@echo "  make mongo       Open Mongo shell (Docker)"
	@echo "  make clean       Cleanup Docker resources"
	@echo ""

# ===============================
# Local Dev (No Docker)
# ===============================
setup:
	python3 -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(VENV)/bin/uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

# ===============================
# Docker
# ===============================
build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

mongo:
	docker exec -it compute-controller-mongo mongosh

clean:
	$(COMPOSE) down -v --rmi all --remove-orphans
