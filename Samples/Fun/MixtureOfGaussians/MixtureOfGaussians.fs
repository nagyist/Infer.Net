
(*
  This corresponds to the C# tutorial MixtureOfGaussians.cs.
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.Lib

open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Maths

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

[<ReflectedDefinition>]
let priors () = 

    let means = [|for i in 0 .. 1 -> random(VectorGaussianFromMeanAndPrecision(VectorFromArray [|0.0; 0.0|], IdentityScaledBy(2,0.01)))|]
    let precs = [|for i in 0 .. 1 -> random(WishartFromShapeAndScale(100.0, IdentityScaledBy(2,0.01)))|]
    let weights = random(Dirichlet([|1.0; 1.0|]))

    (means, precs, weights)

[<ReflectedDefinition>]
let mix(means : Vector[], precs : PositiveDefiniteMatrix[], weights : Vector) = 

    let z = [|for i in 0 .. 300 -> random(Discrete(weights))|]
    let data = [|for zi in z -> random(VectorGaussianFromMeanAndPrecision(means.[zi], precs.[zi]))|]
    data, z

[<ReflectedDefinition>]
let mixtureModel (data : Vector[]) =
    let (means, precs, weights) = priors ()
    let mix, z = mix(means, precs, weights)
    observe(data = mix)
    (means, precs, weights, z)

/////////////////////////////////////////////////
// Data
/////////////////////////////////////////////////


let means = [| Vector.FromArray(2.0, 3.0); Vector.FromArray(7.0, 5.0) |]
let precs = [| new PositiveDefiniteMatrix(array2D [ [ 3.0; 0.2 ]; [ 0.2; 2.0 ] ]);
               new PositiveDefiniteMatrix(array2D [ [ 2.0; 0.4 ]; [ 0.4; 4.0 ] ])|]
let weights = Vector.FromArray(6.0, 4.0)

Rand.Restart(12347)

let data, _ = mix(means, precs, weights)

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

open MicrosoftResearch.Infer.Distributions

let engine = new InferenceEngine(new VariationalMessagePassing())
setEngine engine

let meansV, precsV, weightsV, zV = interpFun4 <@ mixtureModel @> data

// Break symmetry
let zinit = [| for i in 0 .. 300 -> Discrete.PointMass(Rand.Int(2), 2) |]
zV.InitialiseTo(Distribution<int>.Array(zinit)) |> ignore

let meansD = inferVar meansV
let precsD = inferVar precsV
let weightsD = inferVar weightsV

printf "means: \n%O\n" meansD
printf "precs: \n%O\n" precsD
printf "weights: \n%O\n" weightsD
