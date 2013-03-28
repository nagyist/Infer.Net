// Infer.NET Fun QuickStart
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
// We give them names S,I,L to help with Intellisense; when in doubt type "S."
module S = MicrosoftResearch.Infer.Fun.FSharp.Syntax
module I = MicrosoftResearch.Infer.Fun.FSharp.Inference
module CI = MicrosoftResearch.Infer.Fun.Core.Inference

// define a probabilistic model as an F# function returning pair of bools
[<ReflectedDefinition>]
let coins h =
  let c1 = S.random(S.Bernoulli(h))
  let c2 = S.random(S.Bernoulli(h))
  S.observe (c1 || c2)
  c1,c2

// choose a particular inference engine
I.setEngine (new InferenceEngine(new GibbsSampling()))

// apply inference to return a pair of distributions
let d1,d2 = I.inferFun2 <@ coins @> 0.5

