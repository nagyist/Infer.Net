
(*
    Example of computing model evidence.
    See "Computing model evidence" in Infer.NET Fun documentation.
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

[<ReflectedDefinition>]
let model () =
    let evidence = random(Bernoulli(0.5))

    if evidence then
        let coin = random(Bernoulli(0.6))
        observe(coin = true)

    evidence

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

let evidenceModel1D = inferFun1 <@ model @> () 
let evidenceModel1DB = evidenceModel1D :?> MicrosoftResearch.Infer.Distributions.Bernoulli
printf "evidence: \n%O\n" evidenceModel1DB
printf "log odds: \n%O\n" evidenceModel1DB.LogOdds
printf "probability of model being true: \n%O\n" (exp evidenceModel1DB.LogOdds)

