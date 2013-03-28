(*
  The click graph example infers how closely two links are related from how often these links
  are clicked in different contexts.

  We first uniformly draw a prior similarity (similarityAll in the code) between the links from [0,1].
  For each pair of clicks, we then draw from a Boolean variable (sim) denoting whether they
  are similar in this experiment or not.
  If the Boolean is true, we assume that the click probabilities are drawn from the
  same distribution (beta12), otherwise they are drawn from two independent distributions (beta1,beta2).
*)

module ClickGraph

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference

/////////////////////////////////////////////////
// Click-graph model
/////////////////////////////////////////////////

[<ReflectedDefinition>]
let clickGraph (clicks: (bool * bool)[]): float =
  let similarityAll = random(Beta(1.0,1.0))
  let sim = [| for _ in clicks -> 
                 let sim = random(Bernoulli(similarityAll))
                 let beta1,beta2 =
                   if sim then
                     let beta12 = random(Beta(1.0,1.0))
                     (beta12, beta12)
                   else
                     let _beta1 = random(Beta(1.0,1.0))
                     let _beta2 = random(Beta(1.0,1.0))
                     (_beta1,_beta2)
                 let c1 = random(Bernoulli(beta1))
                 let c2 = random(Bernoulli(beta2))
                 (c1,c2) |] 
  observe(sim = clicks)
  similarityAll

let clicks1 = [|true,true; true,false; true,true; false,false; false,false|]
let clicks2 = [|true,true; true,true; true,true; false,false; false,false|]

let sim = inferFun1 <@ clickGraph @> clicks2

printf "Probability of similarity:\n%O\n" sim
