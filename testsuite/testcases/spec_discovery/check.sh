#!/usr/bin/env bash
# spec-discovery: the task never names money.go's RoundCents or its
# round-half-to-even convention — the model has to notice that ledger.go's
# SettlementTotal already rounds cents, go find how, and match it. A
# plausible-looking naive rounding (math.Round's round-half-away-from-zero,
# or plain truncation) compiles fine and passes ordinary cases, so this only
# checks amounts that land exactly on a half-cent tie, where the two
# conventions disagree.
set -euo pipefail
output="$1"

fail() { echo "FAIL: $1"; exit 1; }

[ -f "invoice.go" ] || fail "invoice.go was not created"
grep -q "func InvoiceTotal" invoice.go || fail "invoice.go missing 'func InvoiceTotal'"

go build -o invoice_bin main.go money.go ledger.go invoice.go 2>build_err.txt \
    || fail "compilation failed: $(cat build_err.txt)"

check_case() {
    local expected="$1"; shift
    local got
    got=$(./invoice_bin "$@" 2>run_err.txt) || fail "run failed for args [$*]: $(cat run_err.txt)"
    [ "$got" = "$expected" ] \
        || fail "InvoiceTotal($*) = $got, want $expected (this project rounds half-cent ties to even, not away from zero — see how ledger.go's SettlementTotal rounds via money.go's RoundCents)"
}

# 3.25 + 7.25 = 10.50 -> tie, floor 10 is even -> 10 (naive round-half-up gives 11)
check_case 10 3.25 7.25
# 6.25 + 6.25 = 12.50 -> tie, floor 12 is even -> 12 (naive round-half-up gives 13)
check_case 12 6.25 6.25
# 5.75 + 5.75 = 11.50 -> tie, floor 11 is odd -> 12 (naive truncation gives 11)
check_case 12 5.75 5.75
# 1.10 + 2.20 = 3.30 -> not a tie, sanity check ordinary rounding still works
check_case 3 1.10 2.20

rm -f invoice_bin build_err.txt run_err.txt
echo "PASS"
