// Infer.NET Fun quick reference
//
// Ctrl-Alt-F to start F# Interactive
// Click this code buffer, and Ctrl-A to select the whole buffer in the editor
// Alt-Enter to send the selected region to F# Interactive
// Hover over variable and function names to see their types.

#I @"..\..\Bin" // wherever the .dll files are located
#r @"Infer.Runtime.dll";
#r @"Infer.Compiler.dll";
#r @"Infer.Fun.dll";

open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Maths // Access to Vector and PositiveDefiniteMatrix, etc.

// These are the modules to import to start writing Fun programs.
// We give them names S,I,L,CI to help with Intellisense; when in doubt type "S."
module S = MicrosoftResearch.Infer.Fun.FSharp.Syntax
module I = MicrosoftResearch.Infer.Fun.FSharp.Inference
module L = MicrosoftResearch.Infer.Fun.Lib
module CI = MicrosoftResearch.Infer.Fun.Core.Inference

// For easy reference, we set the inference options.
let engine = new InferenceEngine(new VariationalMessagePassing()) 
// let engine = new InferenceEngine(new ExpectationPropagation()) // default
// let engine = new InferenceEngine(new GibbsSampling())
I.setEngine engine
//I.setVerbose true        // default=false
//I.setShowFactorGraph false  // default=false

////////////////////////////////////////////////////////////////////////////////////////////
// A beginning example: Bayesian linear regression
////////////////////////////////////////////////////////////////////////////////////////////

// We use the 'ReflectedDefinition' attribute to ensure inference can access the syntax tree later.
[<ReflectedDefinition>]
let point x a b invNoise = // generates a 'y' from an 'x' using a linear model with stochastic noise. 
                           // The parameter invNoise is (1 / noise), also called precision.
  let y = a * x + b // deterministic let
  S.random(S.GaussianFromMeanAndPrecision(y, invNoise))

// The parameters of our Bayesian model is the triple a,b,invNoise
// We define our prior uncertainty about a,b,invNoise as the following distribution.
[<ReflectedDefinition>]
let prior () =
  let a = S.random(S.GaussianFromMeanAndPrecision(0.0, 1.0)) // stochastic lets
  let b = S.random(S.GaussianFromMeanAndPrecision(5.0, 0.3))
  let invNoise = S.random(S.GammaFromShapeAndScale(1.0, 1.0))
  a, b, invNoise 

// Models are ordinary F# code, and can be "run forwards" to generate synthetic data.
let nPoints = 10
let aTrue, bTrue, invNoiseTrue = prior ()  // ground truth for this synthetic dataset
let data =  // generate (x,y) pairs from the ground truth
  [| for x in 1.0 .. float(nPoints) -> x, point x aTrue bTrue invNoiseTrue |]

// We can update our prior beliefs using 'observe'.
// (NB 'observe' expressions are analyzed when code is processed during inference,
// but not when code is run directly.)
[<ReflectedDefinition>]
let posterior data =
  let a, b, invNoise = prior ()
  // Fun supports array comprehensions, but not recursion and unbounded loops
  let gendata = [| for (x, _) in data -> x, point x a b invNoise |]
  // we observe equality between two arrays
  S.observe (data = gendata)
  a, b, invNoise

// The 'observe' construct is like an 'assert'.  In normal programming languages, 'assert' 
// lets through only the runs of a program that satisfy the constraint; in Fun, 'observe' lets 
// through only the probability mass from parts of the space that satisfy the constraint.
// We are left with a distribution conditioned on the observation.

[<ReflectedDefinition>]
let posterior2 data = // alternative definition
  let a, b, invNoise = prior ()

  // We can also use array iteration-loops
  for (x,y) in data do S.observe (y = point x a b invNoise) // only need to compare y-values

  // ...or 'range' to get the array of indices.  Useful for traversing multiple arrays simultaneously.
  // S.observe (data = [| for i in S.range(data) -> let x = fst(data.[i]) in x, point x a b invNoise |])

  // ...or directly use integer ranges.  Integer ranges must start at 0.
  // S.observe (data = [| for i in 0 .. data.Length - 1 -> let x = fst(data.[i]) in x, point x a b invNoise |])

  a, b, invNoise


// Models are "run backwards" during inference, via Infer.NET

// The main call.  We use an F# quotation '<@ ...code... @>' to reify the syntax 
// tree, which is then interpreted specially to do inference with Infer.NET.
// Note that the full syntax tree for 'posterior' is available only because we have
// marked its definition and the definitions it refers to with 'ReflectedDefinition'.

let aMarg, bMarg, invNoiseMarg = I.inferFun3 <@ posterior @> data

// The 'inferFun{1,2,3,4}' functions are convenience wrappers around 'inferDynamic', the main 
// entry point into the inference engine.  The posterior is over a 3-tuple of parameters, so 
// we use 'inferFun3', which takes two arguments: (1) a quoted, named function that takes data and 
// returns a 3-tuple, and (2) the actual data.  It returns the three marginal distributions 
// of the posterior.

printfn "true a: %A\ninferred a: %A\n" aTrue aMarg
printfn "true b: %A\ninferred b: %A\n" bTrue bMarg
printfn "true noise (inverse): %A\ninferred noise (inverse): %A\n" invNoiseTrue invNoiseMarg

// The following shows how a dynamic cast to CanGetMean<float> convinces the type-checker that
// we can indeed take the mean of these marginal distributions.  Details on other available
// operations (CanGetVariance, CanGetLogProb, etc.) are documented in Infer.NET.

let aMean = (aMarg :?> CanGetMean<float>).GetMean()
let bMean = (bMarg :?> CanGetMean<float>).GetMean()
printf "mean a: %f\n" aMean
printf "mean b: %f\n" bMean


////////////////////////////////////////////////////////////////////////////////////////////
// Another example: two coins, showing how to pattern-match against CompoundDistribution
////////////////////////////////////////////////////////////////////////////////////////////

type Coins<'T1,'T2> = {Coin1: 'T1; Coin2: 'T2}

[<ReflectedDefinition>]
let coins h =
  let c1 = S.random(S.Bernoulli(h))
  let c2 = S.random(S.Bernoulli(h))
  S.observe (c1 || c2)
  {Coin1=c1; Coin2=c2}

/// Specialisation of inferDynamic to the record type Coins
let inferCoins (e:Quotations.Expr<'TH -> Coins<bool,bool>>) (h:'TH): Coins<Bernoulli,Bernoulli> =
  match I.inferDynamic e h with
  // we pattern match against a record of simple distributions, encoded in a CompoundDistribution
  | CI.Record [|"Coin1"; "Coin2"|] [|CI.Simple(d1); CI.Simple(d2)|] ->
      // and then use downcasts from obj to Bernoulli
      {Coin1=d1 :?> Bernoulli; Coin2=d2 :?> Bernoulli}
  | _ -> failwith "unexpected distribution"

I.setEngine (new InferenceEngine(new GibbsSampling()))

let {Coin1=b1; Coin2=b2} = inferCoins <@ coins @> 0.5

////////////////////////////////////////////////////////////////////////////////////////////
// A more sophisticated example: Multi-dimensional mixture of Gaussians
////////////////////////////////////////////////////////////////////////////////////////////

// In this example we will look at how to use records, how to work with distributions on 
// Vectors, and how to interact more closely with Infer.NET

// Extra functions that Fun knows how to translate to Infer.NET for inference, 
// mainly for working with distributions on Vectors.
//open MicrosoftResearch.Infer.Fun.Lib

// In more complicated models, it is good practice to use records instead of tuples because
// it lets us use named fields instead of needing to remember which tuple position corresponds
// to which parameter.  

type Param = 
 { /// Mean of each Gaussian
   Means: Vector[]
   /// Precision matrix of each Gaussian
   Precs: PositiveDefiniteMatrix[]
   /// Component weights, i.e. probability of being generated by each Gaussian
   Weights: Vector }

// We can now use the 'recordValue.field' syntax.

[<ReflectedDefinition>]
let mixtureOfGaussians (p : Param) = 
  // 'zs' denotes which component each point belongs to
  let zs = [|for i in 0 .. 300 -> S.random(S.Discrete(p.Weights))|]
  // Drawing each point from its respective component
  let xs = [|for z in zs -> S.random(S.VectorGaussianFromMeanAndPrecision(p.Means.[z], p.Precs.[z]))|]
  xs, zs

// Now our prior returns a record.

[<ReflectedDefinition>]
let mogPrior () = // uses several functions from Lib
  let nComponents = 2
  let means = [|for i in 0 .. nComponents-1 -> S.random(S.VectorGaussianFromMeanAndPrecision(L.VectorFromArray [|0.0; 0.0|], L.IdentityScaledBy(2,0.01)))|]
  let precs = [|for i in 0 .. nComponents-1 -> S.random(S.WishartFromShapeAndScale(100.0, L.IdentityScaledBy(2,0.01)))|]
  let weights = S.random(S.Dirichlet([|1.0; 1.0|]))
  {Weights = weights; Precs = precs; Means = means} // any order is fine
  
[<ReflectedDefinition>]
let mogPosterior data =
  let p = mogPrior ()
  let xs, zs = mixtureOfGaussians p
  S.observe (data = xs)
  p, zs

// ground truth
let trueParams = 
 {Means = [| Vector.FromArray(2.0, 3.0); Vector.FromArray(7.0, 5.0) |]
  Precs = [| new PositiveDefiniteMatrix(array2D [ [ 3.0; 0.2 ]; [ 0.2; 2.0 ] ]);
             new PositiveDefiniteMatrix(array2D [ [ 2.0; 0.4 ]; [ 0.4; 4.0 ] ])|]
  Weights = Vector.FromArray(6.0, 4.0)}

// Set the seed for repeatability
Rand.Restart(12347)

// Real data won't be labelled with which component it came from, so we throw it away
let sampledXs, _ = mixtureOfGaussians trueParams

I.setEngine (new InferenceEngine(new VariationalMessagePassing()))

// We need to do two things differently than before.
// (1) use 'interpXXX' instead of 'inferXXX'. 
// (2) use 'XXXdynamic' instead of 'XXXFun{N}'

// (1) Analogous to 'inferDynamic' and 'inferFun{N}', there are functions 'interpDynamic' and
// 'interpFun{N}'.  These functions set up the model in Infer.NET /without/ running inference.  
// This allows us to send extra calls to the Infer.NET API before running inference.  We need
// this to break symmetry in this example.

// (2) The '{infer,interp}Fun{N}' functions only handle models on tuples over scalar and array 
// types.  With records, we need to use '{infer,interp}Dynamic', and unpackage the return value 
// ourself.

// This call sets up the model in Infer.NET.  It returns a 'CompoundVariable', a structure 
// contiaining Infer.NET 'Variable's.  Likewise, 'inferDynamic' returns a 'CompoundDistribution'.
let compoundVar, _ = I.interpDynamic <@ mogPosterior @> sampledXs

// Import some active patterns for destructuring the result
open MicrosoftResearch.Infer.Fun.Core.Inference

// Access to Infer.NET's 'Variable' type
open MicrosoftResearch.Infer.Models

let unpackage result
 : Variable<Vector[]> * Variable<PositiveDefiniteMatrix[]> * Variable<Vector> * Variable<int[]> =
  match result with
  // 'mogPosterior' returns a 2-tuple of a record and an scalar, so we look for 
  // a record of 'CompoundVariable's and a lone 'CompoundVariable'.
  | TupleVar ([| RecordVar paramCVars; zsCVar |]) -> 
    // pattern-match on the map as a list, sorted by field name
    match Map.toList paramCVars, zsCVar with 
    | [ ("Means"  , VarOrVarArray meansVar);
        ("Precs"  , VarOrVarArray precsVar);
        ("Weights", VarOrVarArray weightsVar) ] , VarOrVarArray zsVar 
      // we use VarOrVarArray for scalars and arrays
      -> meansVar, precsVar, weightsVar, zsVar
    | _ -> failwith "impossible"
  | _ -> failwith "impossible"

// We would use the active patterns 'Tuple', 'Record' and 'SimpleOrArray' instead of 
// 'TupleVar', 'RecordVar', and 'VarOrVarArray' if we had used 'inferDynamic' and needed to 
// unpackage a 'CompoundDistribution'.

let meansVar, precsVar, weightsVar, zsVar = unpackage compoundVar

// Now we have access to the individual variables in the model.  We will use 'zsVar' in 
// particular, to break symmetry using a random initialization
let zsInit = [| for i in 0 .. 300 -> Distributions.Discrete.PointMass(Rand.Int(2), 2) |]
zsVar.InitialiseTo(Distribution<int>.Array(zsInit))

// Now we can do inference

let meansMarg = inferVar meansVar
let precsMarg = inferVar precsVar
let weightsMarg = inferVar weightsVar

printf "means: \n%A\n" meansMarg
printf "precs: \n%A\n" precsMarg
printf "weights: \n%A\n" weightsMarg
