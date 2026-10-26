#!/usr/bin/env bash
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
# Builds the Apple Silicon zero-copy Metal payload bench. Objective-C++,
# links Metal.framework directly — no Xcode project, no .metallib, no
# standalone `metal` shader compiler (the kernel is compiled at runtime
# from an embedded MSL string via MTLDevice.newLibraryWithSource).
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Metal zero-copy bench is Apple-platform only." >&2
  exit 1
fi

mkdir -p build
clang++ -std=c++17 -O2 -fobjc-arc \
  -I../include -I. \
  -framework Metal -framework Foundation \
  ../src/core.cpp ../src/aamt_c.cpp aamt_metal.mm aamt_metal_bench.cpp \
  -o build/aamt_metal_bench

echo "built build/aamt_metal_bench"
