
(*
  This corresponds to the C# example TruncatedGaussian.cs.
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.Lib


/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

// external data
let data = [0.0 .. 0.1 .. 1.0]

[<ReflectedDefinition>]
let truncatedGaussianA () =
    let a = data
    let result = [for x in a -> random (GaussianFromMeanAndPrecision(0.0, 1.0))]
    for x, y in List.zip a result do 
        observe (y > x)
    result

// This produces a different result:
[<ReflectedDefinition>]
let truncatedGaussianB () =
    let a = data
    let y = random (GaussianFromMeanAndPrecision(0.0, 1.0))
    let result = [for x in a -> y]
    for x, y in List.zip a result do 
        observe (y > x)
    result
    

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

let compoundAD = inferDynamic <@ truncatedGaussianA @> ()
printf "Distribution A: \n%O\n" compoundAD

let compoundBD = inferDynamic <@ truncatedGaussianB @> ()
printf "Distribution B: \n%O\n" compoundBD
