package main

import "math"

// RoundCents converts a fractional cent amount into an integer number of
// cents using this package's settlement convention: an exact half-cent tie
// rounds to the nearest even cent, not away from zero. Every cents-rounding
// path in this package agrees with it, since a mismatch would make batch
// totals fail to reconcile against their per-line amounts at close-of-day.
func RoundCents(v float64) int64 {
	floor := math.Floor(v)
	diff := v - floor
	switch {
	case diff < 0.5:
		return int64(floor)
	case diff > 0.5:
		return int64(floor) + 1
	default:
		f := int64(floor)
		if f%2 == 0 {
			return f
		}
		return f + 1
	}
}
