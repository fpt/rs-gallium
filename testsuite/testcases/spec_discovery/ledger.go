package main

// SettlementTotal sums a batch of settlement amounts (in cents, given as
// fractional values because upstream fee reports carry sub-cent precision)
// and rounds the result with RoundCents, so a batch total always agrees
// with the per-line rounding applied at close-of-day.
func SettlementTotal(amounts []float64) int64 {
	var sum float64
	for _, a := range amounts {
		sum += a
	}
	return RoundCents(sum)
}
