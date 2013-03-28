module MicrosoftResearch.Infer.Fun.Core.Syntax

open System

///////////////////////////////////////////////////////////////////////////////
/// Expressions

type FunType = | TSimple of System.Type
               | TUnit
               | TRecord of Map<string, FunType> /// records and tuples 
               | TArray of FunType

/// Variables.
/// For non-local variables the name includes the module name.
type vname = string

/// Includes type if known.
type v = vname * FunType option

/// Operations
type UnaryOp = Negate | Not (* | Exp | Logistic *)

type BinaryOp = 
            | Plus | Minus | Mult | Div | Max | Mod
            | Or | And
            | Eq | Neq | Lt | Gt | LtEq | GtEq

val (|UNeedsParens|_|): UnaryOp -> UnaryOp option
val (|BNeedsParens|_|): BinaryOp -> BinaryOp option


/// distribution names
type DistName = 
    | Beta | BetaFromMeanAndVariance 
    | GaussianFromMeanAndPrecision | GaussianFromMeanAndVariance
    | GammaFromShapeAndScale | GammaFromMeanAndVariance | GammaFromShapeAndRate
    | Binomial
    | VectorGaussianFromMeanAndVariance | VectorGaussianFromMeanAndPrecision 
    | Discrete | DiscreteUniform
    | Poisson
    | Bernoulli
    | Dirichlet | DirichletUniform | DirichletSymmetric
    | WishartFromShapeAndScale

type CallInfo =
      /// Call a Fun function. 
      /// Function names are bound to bodies in Context (see below).
    | Internal of string
      /// Call an external function that has an associated factor. 
      /// See Lib.fsi for supported external functions and ways to add arbitrary new ones.
    | External of Reflection.MethodInfo

type constant =
    | B of bool
    | I of int
    | F of float
        /// An opaque object (say, Vector, PositiveDefiniteMatrix).
        /// The reason we require a type annotation is that
        /// o.GetType() returns the runtime type which due
        /// to inheritance may be more specific than the declared type 
        /// (DenseVector vs Vector). This leads to problems in the 
        /// interpreter because T :> T' does not imply Variable<T> :> Variable<T'>.
    | O of obj * System.Type


/// Projection/dereference
type selector = Field of string | Index of e

and e = 
        // Values
        | Unit
            /// Variables.
        | V of v
            /// Constants
        | C of constant
            /// Records.
            /// Tuples are represented as records 
            /// with fields Item1, Item2, ...
        | R of Map<string, e>
            /// Arrays.
            /// Contains the element type for the case the array is empty.
            /// An array element must be either an array (A ...) or an expression of non-array type.
            /// Arrays of comprehensions are not allowed.
        | A of e list * FunType option
            /// Range e = [| 0 .. e - 1 |].
        | Range of e
            /// RangeOf array = [| 0 .. array.Length - 1 |].
            /// In F# code use range(array).
        | RangeOf of e

        // Operations
            /// Projections: a.[i] or a.fieldName.
            /// Nested projections are allowed.
        | P    of e * selector 
        | UOp  of UnaryOp * e 
        | BOp  of BinaryOp * e * e

        // Control
        | If   of e * e * e
        | Call of CallInfo * e list
        | Let  of v * e * e
        | Seq  of e * e
       
        // Probabilistic stuff
        | D of DistName * e list 
        | Observe of e
       
        // Array comprehensions
            /// for v in e1 do e2
            /// The iterator name can be arbitrary
        | Iter of v * e * e 
            /// [|for v in e1 -> e2|]
            /// The iterator name can be arbitrary
        | Map of v * e * e
        | Zip of e * e

        /// annotate(a){e}
        /// In Fun semantics annotate(a){e} is the same as e, but annotations are used to drive Infer.NET correctly. 
        /// Annotations are inserted by program transformations, you should not need to use them.
        | Annotation of annotation * e 


and annotation = 
            /// <summary> switch(e){e'}.</summary>
            /// In Infer.NET interpretation switch(e){e'} is interpreted as Variable.Switch(interp e){interp e'},
            /// thus allowing e to be used as index in e'.
        | Switch of e
            /// <summary> Expression copy.</summary>
            /// In Infer.NET interpretation copy(e) is interpreted as Variable.Copy(interpet e).
            /// Making copies is necessary sometimes because Infer.NET assignment (Variable.SetTo()) 
            /// consumes the argument and makes it unusable, so we need to copy the argument first.
            /// See Transformations.insertCopies for details.
        | Copy
            /// <summary> Value range annotation, valueRange(n){e}.</summary>
            /// In Infer.NET interpretation valueRange(n){e} is interpreted as (interpet e).SetValueRange(Range(n)).
        | ValueRange of int
            /// <summary> Range annotations, ranges(r1, ..., rn){e}.</summary>
            /// Specifies explicitly the ranges of an expression. See Transformations.insertRanges for details.
        | Ranges of v list
            /// <summary> Range cloning annotation, clone(e).</summary>
            /// Tells us that the range expression e needs to be cloned. See Transformations.insertRanges for details.
        | CloneRange


type Body =
        | Lambda of v list * e
        | Value  of e

type Context = Map<vname, Body>

////////////////////////////////////////////////////////////////////////////////
/// Typing

/// Type conversion
val systemTypeToFunType: System.Type -> FunType

val funTypeToSystemType: FunType -> System.Type

val getDistributionType: DistName -> FunType

/// Fills in missing variable types.
/// Assumes a closed expression where all internal calls have been inlined. 
/// See Transformations.inlineContext.
val inferTypes: e -> e

/// Call inferTypes first.
val getType: e -> FunType

////////////////////////////////////////////////////////////////////////////////
/// Traversal

val children: e -> e list

val replaceChildren: e list -> e -> e

/// Apply a function to all children
val descend: (e -> e) -> e -> e

////////////////////////////////////////////////////////////////////////////////
/// Helpers

/// Tuple field names
val proj: int -> string

val distName: string -> DistName

val vname: v -> vname



