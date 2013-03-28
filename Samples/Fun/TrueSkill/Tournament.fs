module Tournament

open TrueSkill

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference

open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Maths

/////////////////////////////////////////////////
// Misc
/////////////////////////////////////////////////

let rand = fun max -> Distributions.Discrete.Uniform(max).Sample()

let randPermuteArray a =
    let n = Array.length a
    let rec aux = function
        | 0 -> a
        | k ->
            let i = rand(k+1)
            let tmp = a.[i]
            a.[i] <- a.[k]
            a.[k] <- tmp
            aux (k-1)
    aux (n-1)

let randPermuteList l = Array.ofList l |> randPermuteArray |> List.ofArray

/////////////////////////////////////////////////
// Tournament
/////////////////////////////////////////////////

(*
  We are using an offline algorithm, that is, after each new game we recompute all the skills
  from scratch taking all the previous games into account.

  We could create an online algorithm by seeding the priors for each inference with posteriors
  from the prevous round. In any case, an offline algorithm is good to have as a baseline 
  for comparison purposes.
*)

(* 
  How many games do we play before we rerun the inference to update the inferred skills.
*)
let gamesPerRound = 10

let play (skills: float[]): player -> player -> outcome = fun p1 p2 ->
    diff2Outcome (performanceDiff skills p1 p2)

(*
  Given the games played so far we run the inference (posterior from above) and then choose the
  players with most similar inferred skills (with preference for those with high skill
  uncertainty)
*)
let nextMatches (games: Game list) (nPlayers: int) (nMatches: int): (player * player) list = 
    //printfn "running inference ... "
    let skills = inferFun1 <@ posterior @> (nPlayers, Array.ofList games)
    let skillsA = Distribution.ToArray<Gaussian[]>(skills)
    let skillsMeans = [| for s in skillsA -> s.GetMean() |]
    let skillsVariances = [| for s in skillsA -> s.GetVariance() |]
    //printfn "inference done"

    let playerDiff (p1: player, p2: player): float =
        abs(skillsMeans.[p1] - skillsMeans.[p2]) / (max skillsVariances.[p1] skillsVariances.[p2])

    let possibleMatches = // Get the closest match for each home player
      [ for p1 in [0 .. nPlayers - 2] do 
            yield (List.sortBy playerDiff 
                [ for p2 in [p1+1 .. nPlayers - 1] do 
                    yield p1, p2 ]) |> List.head
            ]

    // NB: quadratic performance in the number of players, can be optimised
    //printfn "choosing next match ... "
    let sortedMatches = List.sortBy playerDiff possibleMatches
    //printfn "next match chosen"
    Seq.truncate nMatches sortedMatches |> List.ofSeq

let tournament (play: player -> player -> outcome) (nPlayers: int) (nGames: int): Game list =
    
    let rec next (nGames: int) (games: Game list): Game list =
        if nGames = 0 then
            games
        else
            let matches = nextMatches games nPlayers (min nGames gamesPerRound)
            let games' = [for p1, p2 in matches -> p1, p2, play p1 p2]
            printfn "%d games left to play" nGames
            List.iter printGame games'
            next (nGames - games'.Length) (games @ games')

    next nGames []

(*
  What do we want to know once the tournament is over?

  Statistics about the proportion of wins/losses for each player is a good
  indicator of algorithm performance. Should converge to 50%.
*)
let analyze (nPlayers: int) (games: Game list) = 

    let totalImbalance = ref 0.0

    let rec countOutcomes (outcome: outcome) (p: player) (games: Game list): int = 
        match games with
        | (p1, p2, o) :: games' when ((p = p1) || (p = p2)) && (outcome = 0) && (o = 0) -> 1 + countOutcomes outcome p games'
        | (p1, p2, o) :: games' when (p = p1) && (outcome = o) -> 1 + countOutcomes outcome p games'
        | (p1, p2, o) :: games' when (p = p2) && (outcome = -o) -> 1 + countOutcomes outcome p games'
        | _ :: games' -> countOutcomes outcome p games'
        | [] -> 0

    for p in 0 .. nPlayers - 1 do
        let nWins = countOutcomes 1 p games
        let nLosses = countOutcomes -1 p games
        let nDraws = countOutcomes 0 p games

        let balance = float (abs nWins) / float (nWins + nLosses)
        let imbalance = float (abs (nWins - nLosses)) / float (nWins + nLosses + nDraws)
        totalImbalance := !totalImbalance + imbalance

        printfn "player %s: wins: %d, losses: %d, draws: %d, imbalance: %f, balance: %f" (showPlayer p) nWins nLosses nDraws imbalance balance

    // the lower the imbalance the higher the quality of the tournament
    printfn "average tournament imbalance: %f" (!totalImbalance / float nPlayers)

/////////////////////////////////////////////////
// Run the tournament
/////////////////////////////////////////////////

let run () = 

    Rand.Restart(16437)

    let nPlayers = 10
    let nGames = 20
    let players = [| for i in 1 .. nPlayers -> "Player " + string i |]
    let skills = skillsPrior nPlayers

    printfn "\nPart (3): Running the tournament"
    printfn "True skills: \n%A" skills
    let games = tournament (play skills) nPlayers nGames

    printfn "True skills: \n%A" skills
    let skillsD = inferFun1 <@ posterior @> (players.Length, Array.ofList games) 
    let skillsD = Distribution.ToArray<Gaussian[]>(skillsD)

    ////////////////
    // Plot and output results

    let trueSkillsD = [| for skill in skills -> Gaussian.FromMeanAndVariance(skill, 0.0001) |]
    Plot.startNewSheet "Tournament Skills"
    Plot.startNewChart 0.0 0.0 400.0 400.0 "True Skills"
    Plot.plotSkillsAndGames nPlayers players (Array.ofList games) (trueSkillsD)
    Plot.startNewChart 450.0 0.0 400.0 400.0 "Inferred Skills"
    Plot.plotSkillsAndGames nPlayers players (Array.ofList games) (skillsD)

    printf "Inferred skills: \n%O\n" skillsD
    printfn "\nTournament summary:"
    analyze nPlayers games
