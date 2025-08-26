#!/bin/bash
# Format all C/C++ files recursively in the entire codebase
cd "$(dirname "$0")/../.." # Go to repository root
find . \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cxx" \) | xargs clang-format -i
