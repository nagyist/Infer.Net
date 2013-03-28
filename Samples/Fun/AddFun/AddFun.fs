
(*
    This example shows the registerFactor function that allows you to extend
    the set of supported primitives. See Lib.fsi for details.
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.Lib

open MicrosoftResearch.Infer.Maths;
open MicrosoftResearch.Infer.Models;

/////////////////////////////////////////////////
// Functions
/////////////////////////////////////////////////

// Max

let maxFactor : Variable<float> * Variable<float> -> Variable<float> = Variable.Max

registerFactor <@ max: float -> float -> float @> <@ maxFactor @> 

// SumWhere

let sumWhere : bool[] * Vector -> float = fun (bs, v) ->
    let xs = Array.map (function true -> 1.0 | false -> 0.0) bs
    Vector.InnerProduct(v, Vector.FromArray(xs))

let sumWhereFactor : VariableArray<bool> * Variable<Vector> -> Variable<float> = Variable.SumWhere

registerFactor <@ sumWhere @> <@ sumWhereFactor @>

// Identity

let identity : int -> PositiveDefiniteMatrix = PositiveDefiniteMatrix.Identity
let identityFactor : Variable<int> -> Variable<PositiveDefiniteMatrix> = 
    fun input -> lift1 identity input

registerFactor <@ identity @> <@ identityFactor @>

/////////////////////////////////////////////////
// Models
/////////////////////////////////////////////////

[<ReflectedDefinition>]
let maxModel () =
    let x = random(GaussianFromMeanAndVariance(0.0, 1.0))
    let y = random(GaussianFromMeanAndVariance(0.0, 1.0))
    max x y

[<ReflectedDefinition>]
let sumModel () =
    let x = random(VectorGaussianFromMeanAndPrecision(VectorFromArray [|0.0; 0.0; 0.0|], identity(3)))
    let ind = [| true; false; true |]
    sumWhere(ind, x)

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

let maxD = inferFun1 <@ maxModel @> ()
printf "max: \n%O\n" maxD

let sumD = inferFun1 <@ sumModel @> ()
printf "sum: \n%O\n" sumD
