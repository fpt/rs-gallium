package main

import "fmt"

var count int = 0

func increment() {
	count++
}

func main() {
	increment()
	increment()
	increment()
	fmt.Println(count)
}
