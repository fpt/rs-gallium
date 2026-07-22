#!/bin/bash

# Single test runner for the gallium CLI.
# Usage: CLI=path/to/gallium ./testsuite/runner.sh <testcase> <backend>
# Example: ./testsuite/runner.sh capital gemma4
#
# CLI defaults to the gallium_cli.sh adapter, which forwards a TOML backend
# config to the gallium binary via --config and feeds prompts on stdin (a REPL,
# one line per turn). Each non-empty line of prompt.txt becomes a user turn;
# "/quit" is appended to end the session. The test runs with its cwd set to an
# isolated temp dir so the read/glob tools see only the testcase files.

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

script_dir="$(cd "$(dirname "$0")" && pwd)"
proj_root="$(cd "$script_dir/.." && pwd)"

# Default to the gallium_cli.sh adapter (forwards TOML --config to gallium binary).
if [ -z "$CLI" ]; then
    CLI="$script_dir/gallium_cli.sh"
fi
# Resolve to an absolute path (we cd into a temp dir before running).
CLI="$(cd "$(dirname "$CLI")" && pwd)/$(basename "$CLI")"

if [ ! -f "$CLI" ]; then
    echo -e "${RED}Error: CLI binary '$CLI' not found${NC}"
    echo "Build it: cargo build --release -p gallium-agent"
    exit 1
fi

# Load .env from the project root if present (e.g. OPENAI_API_KEY).
if [ -f "$proj_root/.env" ]; then
    set -a; . "$proj_root/.env"; set +a
fi

if [ $# -eq 0 ]; then
    echo -e "${BLUE}🧪 Available Test Cases:${NC}"
    find "$script_dir/testcases" -maxdepth 1 -mindepth 1 -type d | sort | while read -r d; do
        echo "  • $(basename "$d")"
    done
    echo ""
    echo -e "${BLUE}🔧 Available Backends:${NC}"
    find "$script_dir/backends" -maxdepth 1 -name '*.toml' | sort | while read -r f; do
        echo "  • $(basename "$f" .toml)"
    done
    echo ""
    echo "Usage: CLI=path/to/gallium ./runner.sh <testcase> <backend>"
    exit 0
fi

testcase_name="$1"
backend_name="${2:-gemma4}"

testcase_dir="$script_dir/testcases/$testcase_name"
backend_file="$script_dir/backends/$backend_name.toml"

if [ ! -d "$testcase_dir" ]; then
    echo -e "${RED}Error: Testcase '$testcase_name' not found${NC}"; exit 1
fi
if [ ! -f "$backend_file" ]; then
    echo -e "${RED}Error: Backend '$backend_name' not found${NC}"; exit 1
fi
if [ ! -f "$testcase_dir/prompt.txt" ]; then
    echo -e "${RED}Error: $testcase_name/prompt.txt not found${NC}"; exit 1
fi
if [ ! -x "$testcase_dir/check.sh" ]; then
    echo -e "${RED}Error: $testcase_name/check.sh not found or not executable${NC}"; exit 1
fi

echo -e "${BLUE}🧪 Running Single Test${NC}"
echo -e "${CYAN}Testcase: $testcase_name${NC}"
echo -e "${CYAN}Backend:  $backend_name${NC}"
echo -e "${BLUE}Binary:   $CLI${NC}"

output_file="$(mktemp)"
error_file="$(mktemp)"
temp_test_dir="$(mktemp -d)"
echo -e "${YELLOW}🗂️  Temp dir: $temp_test_dir${NC}"

# Copy testcase files (prompt.txt, check.sh, any fixtures) into the temp workdir.
cp -r "$testcase_dir/"* "$temp_test_dir/"
cp "$script_dir/extract_response.sh" "$temp_test_dir/" 2>/dev/null || true
chmod +x "$temp_test_dir/extract_response.sh" 2>/dev/null || true

# Build the stdin stream: every non-empty, non-comment line of prompt.txt is a
# REPL turn, then /quit. ('#'-prefixed lines are comments.)
prompt_stream="$(grep -vE '^\s*#' "$temp_test_dir/prompt.txt" | grep -vE '^\s*$'; echo '/quit')"

echo -e "${CYAN}Running model (cwd=$temp_test_dir)...${NC}"
if ( cd "$temp_test_dir" && echo "$prompt_stream" | "$CLI" --config "$backend_file" ) \
        > "$output_file" 2> "$error_file"; then
    exit_code=0
else
    exit_code=$?
fi

echo ""
echo -e "${BLUE}📋 Assistant output:${NC}"
echo "----------------------------------------"
grep -a '^Assistant:' "$output_file" || cat "$output_file"
echo "----------------------------------------"

if [ $exit_code -ne 0 ]; then
    echo -e "${RED}❌ FAIL: $testcase_name × $backend_name (CLI exit $exit_code)${NC}"
    echo -e "${YELLOW}stderr:${NC}"; tail -20 "$error_file"
    echo -e "${YELLOW}💾 Temp dir preserved: $temp_test_dir${NC}"
    rm -f "$output_file" "$error_file"
    exit 1
fi

echo -e "${YELLOW}🔍 Validating...${NC}"
if ( cd "$temp_test_dir" && TESTSUITE_DIR="$script_dir" ./check.sh "$output_file" "$error_file" ); then
    echo -e "${GREEN}✅ PASS: $testcase_name × $backend_name${NC}"
    rm -rf "$temp_test_dir"; rm -f "$output_file" "$error_file"
    exit 0
else
    echo -e "${RED}❌ FAIL: $testcase_name × $backend_name (check failed)${NC}"
    echo -e "${YELLOW}💾 Temp dir preserved: $temp_test_dir${NC}"
    rm -f "$output_file" "$error_file"
    exit 1
fi
