.PHONY: help install run test clean docker-build docker-run docker-stop

# Default target
help:
	@echo "🎨 Latent Editor - Available Commands:"
	@echo ""
	@echo "📦 Installation:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "🚀 Running:"
	@echo "  make run              Run the application"
	@echo "  make run-check        Check environment and dependencies"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run with Docker Compose"
	@echo "  make docker-stop      Stop Docker containers"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean            Clean cache and temporary files"
	@echo "  make test             Run tests (if available)"
	@echo ""

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

install-dev: install
	@echo "📦 Installing development dependencies..."
	pip install -e .[dev]

# Running
run:
	@echo "🚀 Starting Latent Editor..."
	python run.py

run-check:
	@echo "🔍 Checking environment..."
	python run.py --check-only

# Docker commands
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t latent-editor .

docker-run:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose up -d

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down

# Maintenance
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	rm -rf *.egg-info/ 2>/dev/null || true

test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

# Quick setup for new users
setup: install run-check
	@echo "✅ Setup completed successfully!"
	@echo "🚀 Run 'make run' to start the application" 