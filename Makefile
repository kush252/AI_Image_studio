# Image Processing Platform
.PHONY: help install dev dev-api dev-web build test test-fast bench lint clean docker

VENV := .venv
PY   := $(VENV)/Scripts/python.exe
ifeq ($(OS),)
PY   := $(VENV)/bin/python
endif

help:
	@echo "install    create the venv and install backend + frontend deps"
	@echo "dev        run the API and the Vite dev server together"
	@echo "dev-api    run only the API (http://127.0.0.1:8000)"
	@echo "dev-web    run only the web dev server (http://localhost:5173)"
	@echo "build      build the frontend into backend/app/static"
	@echo "test       run the whole backend test suite"
	@echo "test-fast  skip the slow advisor evaluation"
	@echo "bench      per-operator timings at 1024 px"
	@echo "lint       typecheck the frontend"
	@echo "docker     build and start both services"

install:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r backend/requirements.txt
	cd frontend && npm install

dev-api:
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port 8000

dev-web:
	cd frontend && npm run dev

dev:
	@echo "Run 'make dev-api' and 'make dev-web' in two terminals."

build:
	cd frontend && npm run build

test:
	cd backend && ../$(PY) -m pytest tests -q

test-fast:
	cd backend && ../$(PY) -m pytest tests -q -k "not advisor_eval and not most_cases"

bench:
	cd backend && ../$(PY) -m tests.bench_ops

lint:
	cd frontend && npx tsc --noEmit

clean:
	rm -rf backend/app/static frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .pytest_cache -type d -prune -exec rm -rf {} +

docker:
	docker compose up --build
