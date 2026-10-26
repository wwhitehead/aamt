#!/usr/bin/env bash
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
# Emscripten build of libaamt — the same C ABI (include/aamt/aamt.h), same
# core.cpp, compiled to WASM linear memory. No native code fork: this is
# the identical router the CLI and Python bindings use.
set -euo pipefail
cd "$(dirname "$0")"

command -v emcc >/dev/null || {
  echo "emcc not found — install emscripten (brew install emscripten) or run 'source emsdk_env.sh'" >&2
  exit 1
}

mkdir -p dist

# Memory growth is intentionally off (fixed 128MB instead): a growable
# WebAssembly.Memory.buffer is a "resizable" ArrayBuffer, and Safari's
# TextDecoder.decode() rejects those outright — including on views taken
# from Emscripten's own generated glue (e.g. reading a UTF8 C string), not
# just our own code. A fixed-size buffer sidesteps the whole bug class
# instead of chasing every internal decode() call site.
emcc \
  -O3 -std=c++17 \
  -I../include -I../src \
  ../src/core.cpp ../src/aamt_c.cpp \
  -sMODULARIZE=1 \
  -sEXPORT_ES6=1 \
  -sEXPORT_NAME=createAamtModule \
  -sALLOW_MEMORY_GROWTH=0 \
  -sINITIAL_MEMORY=134217728 \
  -sFILESYSTEM=1 \
  -sEXPORTED_RUNTIME_METHODS=FS,getValue,setValue,stringToUTF8,lengthBytesUTF8,HEAPU8 \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_aamt_open,_aamt_close,_aamt_query,_aamt_query_soft,_aamt_odu_profile,_aamt_node_count,_aamt_record_count,_aamt_content_hash,_aamt_max_depth,_aamt_payload_arena,_aamt_all_records,_aamt_all_records_columnar,_aamt_builder_new,_aamt_builder_add,_aamt_builder_add_mirrored,_aamt_builder_add_ecc,_aamt_builder_add_mirrored_ecc,_aamt_builder_write,_aamt_builder_free,_aamt_verify_chirality,_aamt_recover_record,_aamt_read_verified,_aamt_query_verified,_aamt_query_verified_soft,_aamt_encode_ecc_payload,_aamt_decode_ecc_payload \
  -o dist/aamt.mjs

echo "built dist/aamt.mjs + dist/aamt.wasm"
