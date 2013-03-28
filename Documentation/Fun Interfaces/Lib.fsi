module MicrosoftResearch.Infer.Fun.Lib

open MicrosoftResearch.Infer.Fun.Core.Syntax

open MicrosoftResearch.Infer.Distributions;
open MicrosoftResearch.Infer.Maths;
open MicrosoftResearch.Infer.Models;

open System

open System.Collections.Generic

///////////////////////////////////////////////////////
/// Functions supported in Fun
///////////////////////////////////////////////////////

val DiagonalMatrix: diag: float[] -> PositiveDefiniteMatrix 

val IdentityScaledBy: int * float -> PositiveDefiniteMatrix 

val InnerProduct : Vector * Vector -> float

/// Create a Vector corresponding to the supplied array of floats
val VectorFromArray: float[] -> Vector

/// Gets an array containing (possibly duplicated) items of a source array 
val GetItems: array:'T[] * indices:int[] -> 'T[]

/// Create a T[] random variable array by extracting elements of array at the specified indices, which cannot include duplicates. 
val Subarray: array:'T[] * indices:int[] -> 'T[]

/// Exponentiation
val Exp: float -> float

///////////////////////////////////////////////////////
/// CallInfo for supported functions
///////////////////////////////////////////////////////

/// Example: callInfoForFun <@ InnerProduct @>
/// throws an exception if the function is not supported
val callInfoForFun: Microsoft.FSharp.Quotations.Expr -> CallInfo

///////////////////////////////////////
/// Interface for adding new functions
///////////////////////////////////////

/// Register support for new functions
/// 
/// Call like registerFactor <@ f @> <@ factor @>,
/// where f is the function you would like to use in your Fun code.
/// If f is of type f: ('a -> 'b) then the factor must be of type 
/// factor: (tau('a) -> tau('b)), where 
/// tau('a[]) = VariableArray<tau('a)>,
/// tau('a)   = Variable('a) when 'a is a simple type.
/// See AddFun example for an illustration.
val registerFactor: f: Microsoft.FSharp.Quotations.Expr -> factor: Microsoft.FSharp.Quotations.Expr -> unit

val lift1 : ('a -> 'b) -> (Variable<'a> -> Variable<'b>)
val lift2 : ('a * 'b -> 'c) -> (Variable<'a> * Variable<'b> -> Variable<'c>)

val internal init: unit -> unit
