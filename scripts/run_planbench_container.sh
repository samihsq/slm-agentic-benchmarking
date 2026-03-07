#!/bin/bash
# Helper script to run PlanBench in the Docker container with VAL

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== PlanBench Docker Runner ===${NC}"
echo ""
echo "This script runs PlanBench in a Docker container with VAL pre-installed."
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    exit 1
fi

# Check if LITELLM_API_KEY is set
if [ -z "$LITELLM_API_KEY" ]; then
    echo -e "${YELLOW}Warning: LITELLM_API_KEY is not set. PlanBench needs an API key for Azure models.${NC}"
    echo "Set it with: export LITELLM_API_KEY=your_key"
    echo "Or create a .env file with LITELLM_API_KEY=your_key"
    echo ""
fi

# Build the image
echo "Building Docker image (this may take a few minutes)..."
docker build -f Dockerfile.planbench -t slm-planbench .

# Run with docker-compose (preferred)
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo "Starting container with docker-compose..."
    docker compose -f docker-compose.planbench.yml run --rm planbench python -m pytest tests/benchmarks/test_plan_bench_runner.py -v
else
    echo "Starting container without docker-compose..."
    docker run --rm \
        -v "$(pwd):/workspace" \
        -v "$(pwd)/results:/workspace/results" \
        -e "VAL=/usr/local" \
        -e "LITELLM_API_KEY=${LITELLM_API_KEY}" \
        -e "LITELLM_API_BASE=${LITELLM_API_BASE}" \
        -e "USE_LITELLM=1" \
        slm-planbench \
        python -m pytest tests/benchmarks/test_plan_bench_runner.py -v
fi

echo ""
echo -e "${GREEN}Done! Check results in the results/ directory.${NC}"