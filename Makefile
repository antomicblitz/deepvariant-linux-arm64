# DeepVariant ARM64 — Development & Build Makefile

.PHONY: setup_dev lint type_check test validate quick_validate docker_build benchmark help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Development Setup ---

setup_dev: ## Install dev dependencies via uv
	uv sync --group dev

# --- Code Quality ---

lint: ## Format and lint Python files with ruff
	uv run ruff format scripts/*.py deepvariant/
	uv run ruff check scripts/*.py deepvariant/ --fix

type_check: ## Run pyright type checking
	uv run pyright

test: ## Run tests with pytest
	uv run pytest

validate: lint type_check test ## Full validation (lint + type_check + test)

quick_validate: lint type_check ## Fast validation (lint + type_check, no tests)

# --- Docker Build ---

DOCKER_IMAGE ?= deepvariant-arm64
DOCKER_TAG ?= latest

docker_build: ## Build ARM64 Docker image
	docker buildx build \
		--platform linux/arm64 \
		-f docker/Dockerfile.arm64 \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		.

# --- Benchmarks ---

benchmark: ## Run chr20 benchmark (requires ARM64 + Docker)
	bash scripts/benchmark_arm64.sh
