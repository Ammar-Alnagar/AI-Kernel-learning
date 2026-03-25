#!/usr/bin/env bash
set -euo pipefail

if [[ -d candle_lab ]]; then
  echo "candle_lab already exists"
  exit 0
fi

cargo new candle_lab
cat > candle_lab/Cargo.toml <<'TOML'
[package]
name = "candle_lab"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
rand = "0.8"
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
TOML

mkdir -p candle_lab/src/bin

find . -type f -name '*.rs' | grep -E '^\\./(Module_.*/exercises|Projects/.+)/' | grep -v '/candle_lab/' | while read -r f; do
  rel="${f#./}"
  module=$(echo "$rel" | cut -d/ -f1 | tr '[:upper:]' '[:lower:]')
  base=$(basename "$rel" .rs | tr '[:upper:]' '[:lower:]')
  cp "$f" "candle_lab/src/bin/${module}__${base}.rs"
done

cat > candle_lab/run_solution.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: ./run_solution.sh <bin_name>"
  exit 1
fi
bin="$1"
if [[ "$bin" == *"fill_in"* ]]; then
  echo "Choose a solution bin (without _fill_in)."
  exit 2
fi
cargo run --bin "$bin"
RUN
chmod +x candle_lab/run_solution.sh

ls -1 candle_lab/src/bin | sed 's/\.rs$//' | sort > candle_lab/BIN_LIST_ALL.txt
grep -v 'fill_in$' candle_lab/BIN_LIST_ALL.txt > candle_lab/BIN_LIST_SOLUTIONS.txt

echo "Created candle_lab/ with preloaded bins."
echo "Run a solution: cd candle_lab && ./run_solution.sh module_01_tensor_fundamentals__ex01_tensor_basics"
