
(*
  The classic Monty Hall puzzle:
  http://en.wikipedia.org/wiki/Monty_Hall_problem
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.Lib

/////////////////////////////////////////////////
// A model that doesn't work
/////////////////////////////////////////////////

[<ReflectedDefinition>]
let montyHall () =
    let secret = random(DiscreteUniform(3)) in
    let guess = random(DiscreteUniform(3)) in 
    let info = 
        // this is what the host does
        if secret = guess
        then 
            let x = (random(DiscreteUniform(2)) + 1) + guess in 
            if x >= 3
            then x - 3
            else x
        else 3 - secret - guess
    let new_guess = 3 - guess - info
    guess = secret, new_guess = secret


/////////////////////////////////////////////////
// The model
/////////////////////////////////////////////////

[<ReflectedDefinition>]
let choose (v: float[]): int = random(Discrete(VectorFromArray v))

[<ReflectedDefinition>]
let chooseOther (a: int) (b: int): int = 
    if a = b then
        if a = 0 then choose [| 0.0; 1.0; 1.0 |] else
        if a = 1 then choose [| 1.0; 0.0; 1.0 |] else
                      choose [| 1.0; 0.0; 0.0 |] 
    else
        if a <> 0 && b <> 0 then choose [| 1.0; 0.0; 0.0 |] else
        if a <> 1 && b <> 1 then choose [| 0.0; 1.0; 0.0 |] else
                                 choose [| 0.0; 0.0; 1.0 |]

[<ReflectedDefinition>]
let montyHall2 () =
    let secret = random(DiscreteUniform(3))
    let guess = random(DiscreteUniform(3)) 
    let info = chooseOther secret guess
    let new_guess = chooseOther info guess
    guess = secret, new_guess = secret

/////////////////////////////////////////////////
// Sampling
/////////////////////////////////////////////////

let countTrue: bool list -> int = fun xs -> List.filter (fun x -> x) xs |> List.length

let guessTrue, newGuessTrue = sampleMany montyHall2 10000 () |> List.unzip
let nGoodSamples = List.length newGuessTrue
let nGuessTrue = countTrue guessTrue 
let nNewGuessTrue = countTrue newGuessTrue 

printf "successfull guess proportion: %f\n" (float nGuessTrue / float nGoodSamples)
printf "successfull new guess proportion: %f\n" (float nNewGuessTrue / float nGoodSamples)

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer

// Incorrect results with other engines
setEngine (new InferenceEngine(new GibbsSampling()))

let (guessD, newGuessD) = inferFun2 <@ montyHall2 @> ()
printf "MontyHall distribution: \n%O\n%O\n" guessD newGuessD
