package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	var amounts []float64
	for _, arg := range os.Args[1:] {
		v, err := strconv.ParseFloat(arg, 64)
		if err != nil {
			fmt.Fprintln(os.Stderr, "invalid amount:", arg)
			os.Exit(1)
		}
		amounts = append(amounts, v)
	}
	fmt.Println(InvoiceTotal(amounts))
}
