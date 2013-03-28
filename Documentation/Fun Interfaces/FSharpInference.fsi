module MicrosoftResearch.Infer.Fun.FSharp.Inference

open Microsoft.FSharp.Quotations

open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Models

open MicrosoftResearch.Infer.Fun.Core.Syntax

module Core = MicrosoftResearch.Infer.Fun.Core.Inference

////////
// Types

type CompoundVariable = Core.CompoundVariable
type CompoundDistribution = Core.CompoundDistribution

////////
// Setting inference parameters

val setEngine : InferenceEngine -> unit
val setVerbose : bool -> unit
val setShowFactorGraph : bool -> unit

////////
// Compilation of F# expressions

val getSource: Expr -> Body

/// Get the syntax tree by introspection of F# code
val getCoreSyntax: Expr -> e

/// returns compiled versions of all reflected definitions in the currently loaded assembly
val getAssemblyContext: unit -> Context

////////
/// Inference

/// Takes the pre-RNF type of the expression.
/// The resulting distribution corresponds to pre-RNF type.
val inferCompound: FunType -> CompoundVariable -> CompoundDistribution

/// Returns the pre-RNF type of the expression.
val interpDynamic: Expr<'a -> 'b> -> 'a -> CompoundVariable * FunType
val inferDynamic:  Expr<'a -> 'b> -> 'a -> CompoundDistribution

val inferVar: Variable<'a> -> IDistribution<'a>

val inferFun1: Expr<'a -> 'b> -> 'a -> IDistribution<'b> 
val inferFun2: Expr<'a -> 'b * 'c> -> 'a -> (IDistribution<'b> * IDistribution<'c>) 
val inferFun3: Expr<'a -> 'b * 'c * 'd> -> 'a -> (IDistribution<'b> * IDistribution<'c> * IDistribution<'d>)  
val inferFun4: Expr<'a -> 'b * 'c * 'd * 'e> -> 'a -> (IDistribution<'b> * IDistribution<'c> * IDistribution<'d> * IDistribution<'e>) 

val interpFun1: Expr<'a -> 'b> -> 'a -> Variable<'b> 
val interpFun2: Expr<'a -> 'b * 'c> -> 'a -> (Variable<'b> * Variable<'c>) 
val interpFun3: Expr<'a -> 'b * 'c * 'd> -> 'a -> (Variable<'b> * Variable<'c> * Variable<'d>)  
val interpFun4: Expr<'a -> 'b * 'c * 'd * 'e> -> 'a -> (Variable<'b> * Variable<'c> * Variable<'d> * Variable<'e>) 


