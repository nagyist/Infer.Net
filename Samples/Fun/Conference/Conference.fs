
(* 
    This replicates the conference reviewer model from
    http://blogs.msdn.com/b/infernet_team_blog/archive/2011/09/30/calibrating-reviews-of-conference-submissions.aspx
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.Lib

open MicrosoftResearch.Infer.Models;

type ReviewRaw = {   
    s: int; // index of the submission
    j: int; // index of the reviewer
    e: int; // expertise level
            // 0 = InformedOutsider
            // 1 = Knowledgeable
            // 2 = Expert
    r: float; // recommendation
              // 1.0 = StrongReject
              // 2.0 = Reject
              // 3.0 = WeakReject
              // 4.0 = WeakAccept
              // 5.0 = Accept
              // 6.0 = StrongAccept
}

/////////////////////////////////////////////////
// Functions
/////////////////////////////////////////////////

// See AddFun for more examples of using registerFactor
let moduleInfo = System.Type.GetType "Conference"
let toDoubleFactor : Variable<int> -> Variable<float> = fun x -> lift1 float x
registerFactor <@ float: int -> float @> <@ toDoubleFactor @>

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

// The thresholds between the recommendation levels
let thresholds = [| for i in 1..5 -> (float) i + 0.5 |]

// Parameters
let m_q = 1.0 + 0.5 * (float (Array.length thresholds)) 
let p_q = 1.0
let k_e = 10.0
let beta_e = 10.0
let k_a = 10.0
let beta_a = 10.0

[<ReflectedDefinition>]
let model (reviews:ReviewRaw[], nSubmissions:int, nReviewers:int, nExpertiseLevels:int) = 

  // Priors
  // quality of each paper
  let quality = [|for s in 0 .. nSubmissions - 1 -> random(GaussianFromMeanAndPrecision(m_q, p_q))|] 
  // judgement precision associated with each expertise
  let expertise = [|for e in 0 .. nExpertiseLevels - 1 -> random(GammaFromShapeAndRate(k_e, beta_e))|] 
  // judgement accuracy of each reviewer
  let accuracy = [|for a in 0 .. nReviewers - 1 -> random(GammaFromShapeAndRate(k_a, beta_a))|] 
  // personal thresholds between recommendation levels for each reviewer,
  // depending on reviewer's accuracy
  let theta = [|for a in accuracy -> [|for t in thresholds -> random(GaussianFromMeanAndPrecision(t, a))|] |]

  // Observations
  for {s = rs; e = re; j = rj; r = rr} in Array.toList reviews do
    // the score that the reviewer gives to the paper
    let score = random(GaussianFromMeanAndPrecision(quality.[rs], expertise.[re]))
    for t, threshold in Array.toList (Array.zip theta.[rj] thresholds) do 
        // observe that the reviewer's score compared to reviewer's threshold
        // corresponds to the actual recommendation
        observe ((t < score) = (threshold - 1.0 < rr))

  // Posteriors
  (quality, expertise, accuracy, theta)

/////////////////////////////////////////////////
// Data
/////////////////////////////////////////////////

// Same data as in ConferenceCS - should produce same results
let reviews = 
    [| {j = 0; s = 0; r = 2.0; e = 0};
       {j = 1; s = 0; r = 3.0; e = 1};
       {j = 0; s = 1; r = 5.0; e = 0};
       {j = 1; s = 1; r = 1.0; e = 2};
       {j = 0; s = 2; r = 3.0; e = 0};
       {j = 1; s = 2; r = 6.0; e = 0};|]

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

let qualityD, expertiseD, accuracyD, thetaD = inferFun4 <@ model @> (reviews, 3, 2, 3)

printf "Quality:\n%O\n" qualityD
printf "Thresholds:\n%O\n" thetaD
printf "Expert Precision:\n%O\n" expertiseD
printf "Accuracy:\n%O\n" accuracyD
