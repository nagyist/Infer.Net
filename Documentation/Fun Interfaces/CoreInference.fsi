///////////////////////////////////////////////////////////////////////////////
/// Inference functions
module MicrosoftResearch.Infer.Fun.Core.Inference

open Syntax

open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Models

/// This mirrors the internal representation used by the interpreter (still in RNF-type).
/// You don't need to access this unless you want to tune per-variable inference parameters.
(* This will be removed soon and tuning will be done directly in the model *)
type CompoundVariable = 
    | Unit
    | Var       of MicrosoftResearch.Infer.Models.Variable
    | VarArray  of MicrosoftResearch.Infer.Models.IVariableArray // VariableArray doesn't have a parameterless base class
    | Prod      of Map<string, CompoundVariable> // Represents both record and tuple types. 

val (|RecordVar|_|): CompoundVariable -> Map<string, CompoundVariable> option 
val (|TupleVar|_|):  CompoundVariable -> CompoundVariable[] option 
val (|VarOrVarArray|_|):  CompoundVariable -> Variable<'T> option

type isTuple = bool

type CompoundDistribution =
    | Unit
    | Prod   of isTuple * Map<string, CompoundDistribution>
    | Array  of FunType * CompoundDistribution[] // The type of the array, for the case it is empty
    | Simple of obj // IDistribution does not have a parameterless base class.

    with
      member proj: int -> CompoundDistribution
      member Item: int -> CompoundDistribution with get
      member (?):  string -> CompoundDistribution
      /// For (Simple o) this is o.GetType(), and it distributes over the other constructors in the natural way
      member GetDistType: unit -> FunType 
      (* member simple: obj *)
 
val (|Tuple|_|):  CompoundDistribution -> CompoundDistribution[] option
val (|SimpleOrArray|_|): CompoundDistribution -> obj option
val (|Record|_|): string[] -> CompoundDistribution -> (CompoundDistribution[]) option
val (|Map|_|): 'Key[] -> Map<'Key,'Item> -> ('Item[]) option

////////
/// Setting inference parameters

val setEngine : InferenceEngine -> unit
val setVerbose : bool -> unit
val setShowFactorGraph : bool -> unit

////////
// Inference

/// Takes the pre-RNF type of the expression.
/// The resulting distribution corresponds to pre-RNF type.
val inferCompound: FunType -> CompoundVariable -> CompoundDistribution

/// Returns the pre-RNF type of the expression.
val interpDynamic: Context -> e -> CompoundVariable * FunType
val inferDynamic:  Context -> e -> CompoundDistribution

val inferVar: Variable<'a> -> IDistribution<'a>

val interpFun1:     Context -> e -> Variable<'T>
val interpFun2:     Context -> e -> Variable<'T1> * Variable<'T2>
val interpFun3:     Context -> e -> Variable<'T1> * Variable<'T2> * Variable<'T3>
val interpFun4:     Context -> e -> Variable<'T1> * Variable<'T2> * Variable<'T3> * Variable<'T4>

val inferFun1:     Context -> e -> IDistribution<'T>
val inferFun2:     Context -> e -> IDistribution<'T1> * IDistribution<'T2>
val inferFun3:     Context -> e -> IDistribution<'T1> * IDistribution<'T2> * IDistribution<'T3>
val inferFun4:     Context -> e -> IDistribution<'T1> * IDistribution<'T2> * IDistribution<'T3> * IDistribution<'T4>

