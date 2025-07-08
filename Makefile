.PHONY: build up down logs test clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	docker compose run --rm gateway python -m pytest tests/
	docker compose run --rm retriever python -m pytest tests/
	docker compose run --rm generator python -m pytest tests/

clean:
	docker compose down -v --rmi all
	docker system prune -f