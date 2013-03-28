
(*
    An implementation of Latent Dirichlet Allocation as described the C# example.
    This is a simple version that does not use repeat blocks yet and so is (much) less efficient.
*)

module LDA

open MicrosoftResearch.Infer.Maths
open MicrosoftResearch.Infer.Models
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer

open MicrosoftResearch.Infer.Fun.Core.Inference
open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference

open System

/////////////////////////////////////////////////
// Types
/////////////////////////////////////////////////

type Token =
    | Word of string
    | NotWord of string


type Doc = {
    // Could tokenize the title and use it for learning as well.
    title: string;
    text: Token[];
    topics: Dirichlet option 
}

/// A topic is a distribution over words
type Topic = Dirichlet // IDistribution<Vector>


/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////


(*
    Theta gives the distribution of topics in each document. Such a distribution is a 
    vector whose all components are positive and sum up to 1, in other words, a pie chart.
    The Dirichlet distribution is a distribution over pie charts.

    Similarly phi gives the distritubution of words in each topic.

    Alpha and beta are concentration parameters that govern the distribution of piecharts. 
    Values < 1.0 prefer concentrated distributions with a single prominent component. 
*)

[<ReflectedDefinition>]
let priors (sizeVocab: int) (numTopics: int) (alpha: float) (beta: float) (docs: int[][]) =
    let theta = [| for doc in docs -> random(DirichletSymmetric(numTopics, alpha)) |]
    let phi = [| for i in 0 .. numTopics - 1 -> random(DirichletSymmetric(sizeVocab, beta)) |]
    theta, phi

[<ReflectedDefinition>]
let generateWords (theta: Vector[], phi: Vector[], docs: int[][]): int[][] =
    [| for d in range(docs) -> 
        [| for w in range(docs.[d]) -> 
            let topic = random(Discrete(theta.[d])) in 
            random(Discrete(phi.[topic])) |] |]

[<ReflectedDefinition>]
let model (sizeVocab: int, nTopics: int, alpha: float, beta: float, docs: int[][]) =

    let theta, phi = priors sizeVocab nTopics alpha beta docs
    let docsPrior = generateWords(theta, phi, docs)
    observe(docs = docsPrior)
    theta, phi


/////////////////////////////////////////////////
// Initialisation
/////////////////////////////////////////////////

open MicrosoftResearch.Infer.Distributions

let getInit (numDocs: int) (numTopics: int): IDistribution<Vector[]> =
    let baseVal = 1.0 / float numTopics

    let initTheta = 
        [| for i in 0 .. numDocs - 1 ->
            // Choose a random topic
            let v = Vector.Zero(numTopics)
            let topic = Rand.Int(numTopics)
            v.[topic] <- 1.0
            Dirichlet.PointMass(v)
        |]

    Distribution<Vector>.Array(initTheta)

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

/// Return the distribution of topics in documents,
/// the distribution of words in topics,
/// and model evidence
let infer (sizeVocab: int) (numTopics: int) (alpha: float) (beta: float) (docs: int[][]): Dirichlet[] * Dirichlet[] * float = 

    let engine = new InferenceEngine(new VariationalMessagePassing())

    engine.NumberOfIterations <- 10

    setEngine engine

    let evidenceVar = Variable.Bernoulli(0.5)
    let evidenceBlock = Variable.If(evidenceVar)
    let thetaVar, phiVar = interpFun2 <@ model @> (sizeVocab, numTopics, alpha, beta, docs)
    evidenceBlock.CloseBlock()

    // Initialisation of variables
    thetaVar.SetSparsity(Sparsity.Dense)
    phiVar.SetSparsity(Sparsity.ApproximateWithTolerance(0.00000000001))
    thetaVar.InitialiseTo(getInit docs.Length numTopics) |> ignore

    engine.OptimiseForVariables <- [| thetaVar; phiVar; evidenceVar |]

    let postTheta, postPhi = inferVar thetaVar, inferVar phiVar
    let postEvidence = inferVar evidenceVar :?> Bernoulli

    printf "Evidence: %A\n" postEvidence

    Distribution.ToArray<Dirichlet[]>(postTheta), Distribution.ToArray<Dirichlet[]>(postPhi), postEvidence.LogOdds

