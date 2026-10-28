# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
cmake_minimum_required(VERSION 3.16)
project(libaamt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

add_library(aamt SHARED src/core.cpp src/aamt_c.cpp)
target_include_directories(aamt PUBLIC include PRIVATE src)

add_executable(aamt_cli tools/aamt_cli.cpp)
target_link_libraries(aamt_cli PRIVATE aamt)
set_target_properties(aamt_cli PROPERTIES OUTPUT_NAME aamt)

enable_testing()
add_executable(aamt_roundtrip tests/roundtrip.cpp)
target_link_libraries(aamt_roundtrip PRIVATE aamt)
add_test(NAME roundtrip COMMAND aamt_roundtrip)

add_executable(aamt_rs_gf16_test tests/test_rs_gf16.cpp)
target_include_directories(aamt_rs_gf16_test PRIVATE src)
add_test(NAME rs_gf16 COMMAND aamt_rs_gf16_test)
add_executable(aamt_selfheal_test tests/test_selfheal.cpp)
target_link_libraries(aamt_selfheal_test PRIVATE aamt)
add_test(NAME selfheal COMMAND aamt_selfheal_test)
