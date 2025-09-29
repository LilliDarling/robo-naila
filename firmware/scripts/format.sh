#!/bin/bash
# Format all C/C++ files recursively in the entire codebase

# Load common functions
source "$(dirname "$0")/common.sh"

print_step "Formatting all C/C++ files..."

# Go to repository root
cd "$(dirname "$0")/../.."

print_info "Searching for C/C++ files to format..."
find . \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cxx" \) | xargs clang-format -i

print_success "All C/C++ files formatted!"
