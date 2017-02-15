package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// Nicely format matrix output.
func pprint(s string, m mat64.Matrix) {
	formatter := mat64.Formatted(m, mat64.Squeeze())
	fmt.Printf("%s\n%v\n\n\n", s, formatter)
}

// Broadcast (1-x) for all elements of the matrix
func minusOne(m *mat64.Dense) *mat64.Dense {
	tmp := mat64.NewDense(0, 0, nil)
	tmp.Clone(m)

	tmp.Apply(func(_, _ int, v float64) float64 {
		return (1 - v)
	}, tmp)

	return tmp
}

// Calculate sigmoid function and its derivitive
func nonlin(x *mat64.Dense, deriv bool) *mat64.Dense {
	tmp := mat64.NewDense(0, 0, nil)
	tmp.Clone(x)

	if deriv == true {
		tmp.MulElem(tmp, minusOne(tmp))
	} else {
		tmp.Apply(func(_, _ int, v float64) float64 {
			return 1 / (1 + math.Exp(-v))
		}, tmp)
	}
	return tmp
}

func main() {

	rand.Seed(12345)

	// Initial Inputs to NN
	X := mat64.NewDense(4, 3, []float64{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1})
	pprint("X is", X)

	// Output we are trying to learn.
	y := mat64.NewDense(4, 1, []float64{0, 0, 1, 1})
	pprint("y is", y)

	// Initial weights of the synapses
	syn0 := mat64.NewDense(3, 1, []float64{-0.16595599, 0.44064899, -0.99977125})
	//syn0 := mat64.NewDense(3, 1, nil)
	//syn0.Apply(func(i, j int, v float64) float64 {
	//	return ((rand.NormFloat64() * 2) - 1)
	//}, syn0)
	pprint("syn0 is", syn0)

	var l0, l1 *mat64.Dense
	for i := 0; i < 10000; i++ {
		// Foward propogation
		l0 = X
		l0a := mat64.NewDense(0, 0, nil)
		l0a.Mul(l0, syn0)
		l1 = nonlin(l0a, false)

		// How much did we miss?
		l1Error := mat64.NewDense(0, 0, nil)
		l1Error.Sub(y, l1)

		// Multiply how much we missed by the slope of
		// sigmoid function at the values of l1
		l1Delta := mat64.NewDense(0, 0, nil)
		l1Delta.MulElem(l1Error, nonlin(l1, true))

		// Update the weights of the synapses.
		t := mat64.NewDense(0, 0, nil)
		t.Mul(l0.T(), l1Delta)
		syn0.Add(syn0, t)
	}

	pprint("Output After Training:", l1)

	// Expected Results:
	// [[ 0.00966449]
	//  [ 0.00786506]
	//  [ 0.99358898]
	//  [ 0.99211957]]
}
