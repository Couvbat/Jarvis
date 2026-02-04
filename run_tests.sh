#!/bin/bash

# Test runner script for Jarvis
# This script runs the test suite with various options

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Jarvis Test Suite ===${NC}\n"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install it with: pip install -r requirements.txt"
    exit 1
fi

# Default options
TEST_TYPE="all"
COVERAGE=true
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)
            TEST_TYPE="unit"
            shift
            ;;
        -i|--integration)
            TEST_TYPE="integration"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -u, --unit          Run only unit tests"
            echo "  -i, --integration   Run only integration tests"
            echo "  --no-coverage       Skip coverage report"
            echo "  -v, --verbose       Verbose output"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests with coverage"
            echo "  ./run_tests.sh -u                 # Run only unit tests"
            echo "  ./run_tests.sh -v --no-coverage   # Verbose without coverage"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/"

# Add marker filter
if [ "$TEST_TYPE" = "unit" ]; then
    PYTEST_CMD="$PYTEST_CMD -m unit"
    echo -e "${YELLOW}Running unit tests only${NC}\n"
elif [ "$TEST_TYPE" = "integration" ]; then
    PYTEST_CMD="$PYTEST_CMD -m integration"
    echo -e "${YELLOW}Running integration tests only${NC}\n"
else
    echo -e "${YELLOW}Running all tests${NC}\n"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=term-missing --cov-report=html"
fi

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Run tests
echo -e "${GREEN}Executing: $PYTEST_CMD${NC}\n"
$PYTEST_CMD

# Show results
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo -e "\n${GREEN}Coverage report saved to: htmlcov/index.html${NC}"
        echo "Open it with: xdg-open htmlcov/index.html"
    fi
else
    echo -e "\n${RED}✗ Some tests failed${NC}"
    exit 1
fi
