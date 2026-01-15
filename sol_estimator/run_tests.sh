#!/bin/bash
# Run SOL estimator unit tests

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running SOL Estimator Unit Tests"
echo "================================="

cd "$SCRIPT_DIR"

# Run pytest with verbose output
python3 -m pytest tests/ -v --tb=short "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "All tests passed!"
else
    echo ""
    echo "Some tests failed (exit code: $exit_code)"
fi

exit $exit_code
