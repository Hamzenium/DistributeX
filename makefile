# Makefile for managing the Flower + TensorFlow federated learning project

VENV_NAME=venv
PYTHON=$(VENV_NAME)/bin/python
PIP=$(VENV_NAME)/bin/pip

# Create and initialize a virtual environment
venv:
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

# Run the Flower simulation
run:
	venv/bin/python main.py

# Reinstall the entire environment from scratch
reinstall:
	rm -rf $(VENV_NAME)
	make venv

# Downgrade NumPy to avoid version 2.x compatibility issues
fix-numpy:
	$(PIP) install "numpy<2"
