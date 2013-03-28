
(*
  TrueSkill model. Given a set of game outcomes between various players, infer the player skills.
*)

module TrueSkill

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference

open MicrosoftResearch.Infer

open MicrosoftResearch.Infer.Maths


/////////////////////////////////////////////////
// Types
/////////////////////////////////////////////////

type player = int
(*
   1: first player wins, 
  -1: second player wins, 
   0: a draw 
*)
type outcome = int32
type Game = player * player * outcome

// ---------------------------------------------------------------------------
// Part (1): prior for TrueSkill
// ---------------------------------------------------------------------------

[<ReflectedDefinition>]
let drawMargin = 10.0

[<ReflectedDefinition>]
let sigma2 = (25.0/3.0) * (25.0/3.0)

[<ReflectedDefinition>]
let beta2 = (25.0/3.0) * (25.0/3.0)

(*
  Return the difference in performances of two players in a game against each other.
*)
[<ReflectedDefinition>]
let performanceDiff (skills: float[]) (p1: player) (p2: player): float = 
    let perf1 = random (GaussianFromMeanAndPrecision(skills.[p1], 1.0/beta2))
    let perf2 = random (GaussianFromMeanAndPrecision(skills.[p2], 1.0/beta2))
    perf1 - perf2

// [<ReflectedDefinition>]
let diff2Outcome (diff: float): int =
    if diff > drawMargin then 1 else
    if diff < -drawMargin then -1
    else 0

[<ReflectedDefinition>]
let skillsPrior (nPlayers: int): float[] = 
    [| for p in 0 .. nPlayers - 1 -> random (GaussianFromMeanAndPrecision(25.0,1.0/sigma2)) |]

[<ReflectedDefinition>]
let prior (nPlayers: int) (matches: (int*int)[]) =
  let skills = skillsPrior nPlayers
  let results = [| for (p1, p2) in Array.toList matches -> performanceDiff skills p1 p2|]
  skills, results

  
// ---------------------------------------------------------------------------
// Part (2): conditioning on concrete data
// ---------------------------------------------------------------------------

[<ReflectedDefinition>]
let posterior (nPlayers: int, games: Game[]) =
  let matches = [| for (p1, p2, _) in games -> (p1, p2) |]
  let skills, synthetic_results = prior nPlayers matches
  for (diff, (_, _, outcome)) in Array.toList (Array.zip synthetic_results games) do
    if outcome = 1 then
      observe (diff > drawMargin)
    else if outcome = -1 then
      observe (diff < -drawMargin)
    else
      observe ((diff >= -drawMargin) && (diff <= drawMargin))
  skills

/////////////////////////////////////////////////
// Pretty printing
/////////////////////////////////////////////////

let showPlayer id =
   sprintf "%d" (id + 1)  // id+1

let showGame: Game -> string = fun (p1, p2, outcome) ->
  if outcome = 1 then sprintf "Player %s beats %s" (showPlayer p1) (showPlayer p2) else
  if outcome = -1 then sprintf "Player %s beaten by %s" (showPlayer p1) (showPlayer p2) else
  sprintf "Players %s and %s drew" (showPlayer p1) (showPlayer p2)

let printGame (g: Game) = printfn "%s" (showGame g)
