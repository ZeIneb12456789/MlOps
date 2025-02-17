# Makefile for automating project tasks

# Variables
PYTHON = python3
PIP = pip3

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Format code using black
format:
	black .

# Check code security using bandit
security:
	bandit -r .

# Prepare data
prepare:
	$(PYTHON) main.py prepare

# Train the model
train:
	$(PYTHON) main.py train
# Evaluate the model
evaluate:
	$(PYTHON) main.py evaluate
run_api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000


# Run all CI steps (format, lint, security, prepare, train, test)
ci: format security prepare train  evaluate run_api

# Clean up temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	rm -rf *.pyo
	rm -rf *.pyd
	rm -rf *.so
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
