#!/usr/bin/env bash
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
# Build (and optionally test) libaamt: ./build.sh [test]
set -euo pipefail
cd "$(dirname "$0")"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --parallel

if [[ "${1:-}" == "test" ]]; then
  ctest --test-dir build --output-on-failure
fi
