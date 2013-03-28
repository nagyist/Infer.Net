(*
  This corresponds to the C# ClinicalTrial example.
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

// This models the probability that the drug is effective.
[<ReflectedDefinition>]
let clinicalTrialEffective (controlGroup, treatedGroup) = 
    let isEffective = random(Bernoulli(0.5))
    let probIfTreated, probIfControl = 
      if isEffective then
        (random(Beta(1.0, 1.0)), random(Beta(1.0, 1.0)))
      else
        let p = random(Beta(1.0, 1.0)) in (p, p)

    observe(controlGroup = [|for c in controlGroup -> random(Bernoulli(probIfControl))|])
    observe(treatedGroup = [|for c in treatedGroup -> random(Bernoulli(probIfTreated))|])

    isEffective, probIfTreated, probIfControl

// This models the effectiveness in the treated and the control GIVEN that the drug is effective.
[<ReflectedDefinition>]
let clinicalTrialProbabilities (controlGroup, treatedGroup) = 
    let isEffective, probIfTreated, probIfControl = clinicalTrialEffective (controlGroup, treatedGroup)
    observe(isEffective)
    probIfTreated, probIfControl

/////////////////////////////////////////////////
// Data
/////////////////////////////////////////////////

let controlGroup = [|false; false; true; false; false|]
let treatedGroup = [|true; false; true; true; true|]

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

let isEffectiveD, _, _             = inferFun3 <@ clinicalTrialEffective @> (controlGroup, treatedGroup)
let probIfTreatedD, probIfControlD = inferFun2 <@ clinicalTrialProbabilities @> (controlGroup, treatedGroup)

printf "Probability the drug is effective: %O\n" isEffectiveD
printf "Effectiveness in the treated group: %O\n" probIfTreatedD
printf "Effectiveness in the control group: %O\n" probIfControlD

