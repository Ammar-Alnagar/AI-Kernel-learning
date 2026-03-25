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
# Fill these with your chosen versions.
candle-core = "0.0"
candle-nn = "0.0"
candle-transformers = "0.0"
TOML

mkdir -p candle_lab/src/bin

echo "Created candle_lab/ with placeholder Candle dependency versions."
echo "Update candle versions in candle_lab/Cargo.toml, then copy exercises into src/bin/ and run cargo." 
