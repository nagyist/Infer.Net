module Chess


open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Maths

open TrueSkill
open Input
open Plot

open MicrosoftResearch.Infer.Fun.FSharp.Inference

open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Models

// setVerbose true  // uncomment to see the corresponding C# code
// setShowFactorGraph false

let data = @"../../Data/" 

let diff (skills: float[]): player -> player -> outcome = fun p1 p2 ->
    diff2Outcome (performanceDiff skills p1 p2)

let runSynthetic () =
    let nPlayers = 5
    let players = [| "A"; "B"; "C"; "D"; "E"|]

    ////////////////
    // Create synthetic data by playing all combinations
    let matches = Array.concat [| for i in 0 .. nPlayers - 1 -> [| for j in i + 1 .. nPlayers - 1 -> (i, j) |] |] 
    let trueSkills, outcomes = prior nPlayers matches
    let games = [| for ((p1, p2), outcome) in Array.zip matches outcomes -> (p1, p2, diff2Outcome outcome) |]
    // convert synthetic skills to gaussians, choose small variance just for the purpose of plotting
    let trueSkillsD = [| for skill in trueSkills -> Gaussian.FromMeanAndVariance(skill, 0.0001) |]

    ////////////////
    // Infer skills based on results
    let posteriorSkillsD = inferFun1 <@ posterior @> (nPlayers, games)

    ////////////////
    // Plot and output results
    startNewSheet "Synthetic Games"
    startNewChart 0.0 0.0 400.0 400.0 "True Skills"
    Plot.plotSkillsAndGames nPlayers players games (trueSkillsD)
    startNewChart 450.0 0.0 400.0 400.0 "Inferred Skills"
    Plot.plotSkillsAndGames nPlayers players games (Distribution.ToArray<Gaussian[]>(posteriorSkillsD))

    printf "Inferred skills: \n%O\n" posteriorSkillsD
   

let run3Players () = 

    // ---------------------------------------------------------------------------
    // Synthetic data
    // ---------------------------------------------------------------------------

    let chess_players, chess_games = load (data + "3PlayersPlayers.txt") (data + "3PlayersGames.txt") 

    // forward execution of prior to obtain samples of skills and results
    let chess_matches = [| for (p1,p2,_) in chess_games -> (p1,p2) |]
    let synthetic_skills, synthetic_results = prior chess_players.Length chess_matches

    let firstPlayers = min 10 chess_players.Length
    let firstResults = min 10 chess_matches.Length

    do printfn "Part (1): Synthetic data generated from the prior distributions"
    do printfn "First few synthetic skills:"
    do for p in 0..(firstPlayers - 1) do printfn "Player %s has skill %O" chess_players.[p] synthetic_skills.[p]
    do printfn "First few synthetic results:"
    do for i in 0..(firstResults - 1) do printGame (fst(chess_matches.[i]), snd(chess_matches.[i]), diff2Outcome(synthetic_results.[i]))
    
    // ---------------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------------
    
    do printfn "\n\nnow for inference..."

    printfn "\nPart (2): Posterior distributions inferred from concrete results"
    printfn "First 10 concrete results:"
    for i in 0 .. (firstResults - 1) do printGame chess_games.[i]
    
    // backward execution, conditioned on concrete results, to obtain distributions of skills
    let skillsD = inferFun1 <@ posterior @> (chess_players.Length, chess_games)

    startNewSheet "3 Players"
    startNewChart 0.0 0.0 600.0 600.0 "Skills and Games"
    plotSkillsAndGames firstPlayers chess_players chess_games (Distribution.ToArray<Gaussian[]>(skillsD))
    startNewChart 600.0 0.0 400.0 400.0 "Player Skills"
    plotCurves firstPlayers chess_players (Distribution.ToArray<Gaussian[]>(skillsD))

    printf "Inferred skills: \n%O\n" skillsD


let run() =
    // run3Players()
    // Rand.Restart(16437)
    runSynthetic()
    // closeExcel()

