.PHONY: help build up down logs eval clean test

help:
	@echo "Available commands:"
	@echo "  make build    - Build docker images"
	@echo "  make up       - Start all services (API + UI) in detached mode"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - Tail logs for all services"
	@echo "  make eval     - Run the LLM-as-a-judge evaluation suite via Docker"
	@echo "  make test     - Run the integration tests against the live API"
	@echo "  make clean    - Remove docker images, volumes, and temporary files"

build:
	docker compose build

up:
	docker compose up -d
	@echo "Waiting for services to become healthy..."
	@sleep 5
	docker compose ps

down:
	docker compose down

logs:
	docker compose logs -f

eval:
	docker compose -f docker-compose.eval.yml up --build
	@echo "Evaluation complete. Check test_reports/ for results."

test:
	@echo "Running integration tests..."
	docker compose exec -T api python tests/test_report.py

clean:
	docker compose down -v --rmi all
	docker compose -f docker-compose.eval.yml down -v --rmi all
	rm -rf data/tmp/*
	@echo "Cleanup complete."
