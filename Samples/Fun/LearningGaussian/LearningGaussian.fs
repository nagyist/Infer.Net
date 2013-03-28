
(*
  This corresponds to the C# tutorial LearningAGaussian.cs.
*)

module LearningGaussian

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.Lib

open MicrosoftResearch.Infer.Maths;

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

// The following are two equivalent models, which should return the same result. 

[<ReflectedDefinition>]
let learningGaussianA data =
    let mean = random (GaussianFromMeanAndPrecision(0.0, 100.0)) 
    let prec = random (GammaFromShapeAndScale(1.0, 1.0))
    for x in data do
        let y = random (GaussianFromMeanAndPrecision(mean, prec))
        observe (y = x)
    mean, prec

[<ReflectedDefinition>]
let learningGaussianB data =
    let mean = random (GaussianFromMeanAndPrecision(0.0, 100.0))
    let prec = random (GammaFromShapeAndScale(1.0, 1.0))
    observe (data = [| for x in data -> random (GaussianFromMeanAndPrecision(mean, prec)) |])
    mean, prec

/////////////////////////////////////////////////
// Data
/////////////////////////////////////////////////

let gaussianData = [|for x in 1..100 -> Rand.Normal(0.0, 1.0)|]

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

let meanA, precA = inferFun2 <@ learningGaussianA @> gaussianData
printf "meanA: \n%O\n" meanA
printf "precA: \n%O\n" precA

let meanB, precB = inferFun2 <@ learningGaussianB @> gaussianData
printf "meanB: \n%O\n" meanB
printf "precB: \n%O\n" precB
