#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./run_solution.sh <bin_name>"
  echo "Example: ./run_solution.sh module_01_tensor_fundamentals__ex01_tensor_basics"
  exit 1
fi

bin="$1"
if [[ "$bin" == *"fill_in"* ]]; then
  echo "Refusing to run fill_in bin: $bin"
  echo "Choose a solution bin (without _fill_in)."
  exit 2
fi

cargo run --bin "$bin"
