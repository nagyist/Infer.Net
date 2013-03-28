
(*
  Shows inference of the posterior distribution of two coins,
  given that at least one of them is tails.
*)

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

open MicrosoftResearch.Infer.Fun.FSharp.Syntax

[<ReflectedDefinition>]
let coins () =
    let c1 = random (Bernoulli(0.5))
    let c2 = random (Bernoulli(0.5))
    let bothHeads = c1 && c2
    observe (bothHeads = false)
    c1, c2, bothHeads

/////////////////////////////////////////////////
// Sampling
/////////////////////////////////////////////////

// Sampling does not take observations into account.
printf "Sample: %O\n" (coins ())

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

open MicrosoftResearch.Infer.Fun.FSharp.Inference

let (c1D,c2D,bothD) = inferFun3 <@ coins @> ()
printf "coinsD: \n%O\n%O\n%O\n" c1D c2D bothD

