
(*
  This corresponds to the C# tutorial BayesPointMachineExample.cs.
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.Lib

open MicrosoftResearch.Infer.Maths;

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

[<ReflectedDefinition>]
let BayesPointMachine (features : Vector[]) (noise:float) (weights:Vector) : bool[] = 
    [|for v in features -> 
        (random(GaussianFromMeanAndVariance(InnerProduct(v, weights), noise)) > 0.0)|]

// Compute the normal vector to a plane through the origin separating the classes
[<ReflectedDefinition>]
let TrainBayesPointMachine features outcomes noise = 
    let priorMeans = VectorFromArray [|0.0; 0.0; 0.0|]
    let priorCovariance = DiagonalMatrix [|1.0; 1.0; 1.0|]
    let weights = random (VectorGaussianFromMeanAndVariance(priorMeans, priorCovariance))
    observe (outcomes = BayesPointMachine features noise weights)
    weights 

[<ReflectedDefinition>]
let trainedWeights = fun (trainingVectors, trainingOutcomes) ->
    TrainBayesPointMachine trainingVectors trainingOutcomes 0.1

[<ReflectedDefinition>]
let modelledOutcomes(trainingVectors, trainingOutcomes, testVectors) =
  BayesPointMachine testVectors 0.1 (trainedWeights(trainingVectors, trainingOutcomes))

/////////////////////////////////////////////////
// Data
/////////////////////////////////////////////////

let featureVector : float * float -> Vector = function (x, y) -> VectorFromArray [|x;y;1.0|]

let trainingFeatures = [|(63.0, 38.0); (16.0,23.0); (28.0,40.0); (55.0,27.0); (22.0,18.0); (20.0,40.0)|]
let trainingVectors = Array.map featureVector trainingFeatures
let trainingOutcomes = [| true; false; true; true; false; false |]
let testFeatures = [|(58.0,36.0); (18.0,24.0); (22.0,37.0);|]
let testVectors = Array.map featureVector testFeatures


/////////////////////////////////////////////////
// Plotting
/////////////////////////////////////////////////

open Microsoft.Office.Interop.Excel

let createChart(): Chart =
    let xl = ApplicationClass()
    xl.Visible <- true

    let book = xl.Workbooks.Add() :> _Workbook
    let sheet = book.Sheets.Add() :?> _Worksheet

    let chartObjects = sheet.ChartObjects() :?> ChartObjects
    let chartObject  = chartObjects.Add(100.0, 50.0, 400.0, 400.0)
    chartObject.Chart

let formatChart (chart: Chart) = 
    chart.ApplyLayout(4)

    let xAxis = chart.Axes(XlAxisType.xlCategory) :?> Axis
    let yAxis = chart.Axes(XlAxisType.xlValue) :?> Axis
    xAxis.HasTitle <- true
    xAxis.AxisTitle.Text <- "Income"
    yAxis.HasTitle <- true
    yAxis.AxisTitle.Text <- "Age"


let addPoints (chart: Chart) (data: (float*float)[]) (title: string) =
    let series = chart.SeriesCollection() :?> SeriesCollection
    let dataSeries = series.NewSeries()
    let dataX, dataY = Array.unzip data
    dataSeries.XValues <- dataX
    dataSeries.Values <- dataY
    dataSeries.ChartType <- XlChartType.xlXYScatter
    dataSeries.Name <- title

let addSeparator (chart: Chart) (weights: Vector) = 
    let a, b, c = weights.[0], weights.[1], weights.[2]

    let xAxis = chart.Axes(XlAxisType.xlCategory) :?> Axis
    xAxis.MinimumScaleIsAuto <- false
    xAxis.MaximumScaleIsAuto <- false
    let xMin = xAxis.MinimumScale
    let xMax = xAxis.MaximumScale

    let yAxis = chart.Axes(XlAxisType.xlValue) :?> Axis
    yAxis.MinimumScaleIsAuto <- false
    yAxis.MaximumScaleIsAuto <- false

    let series = chart.SeriesCollection() :?> SeriesCollection
    let sepSeries = series.NewSeries()
    sepSeries.XValues <- [| xMin; xMax |]
    sepSeries.Values <- [| -(a * xMin + c) / b; -(a * xMax + c) / b |]
    sepSeries.ChartType <- XlChartType.xlXYScatterLinesNoMarkers
    sepSeries.Name <- "linear separator"


/////////////////////////////////////////////////
// Misc
/////////////////////////////////////////////////

let filteri: (int -> bool) -> 'T [] -> 'T [] = fun f -> 
    let rec doFilter i = function
        | x :: xs -> let xs' = doFilter (i + 1) xs in if f i then x :: xs' else xs'
        | [] -> []
    
    in fun xs -> Array.ofList (doFilter 0 (List.ofArray xs))

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

open MicrosoftResearch.Infer.Distributions

let trainedWeightsD = inferFun1 <@ trainedWeights @> (trainingVectors, trainingOutcomes)
let modelledOutcomesD = inferFun1 <@ modelledOutcomes @> (trainingVectors, trainingOutcomes, testVectors)

printf "trainedWeights: \n%A\n" trainedWeightsD
printf "modelledOutcomes: \n%A\n" modelledOutcomesD

let trainedWeightsM = (trainedWeightsD :?> CanGetMean<Vector>).GetMean()

let trueTrainingFeatures = filteri (fun i -> trainingOutcomes.[i]) trainingFeatures
let falseTrainingFeatures = filteri (fun i -> not trainingOutcomes.[i]) trainingFeatures

let chart = createChart()
addPoints chart trueTrainingFeatures "did buy"
addPoints chart falseTrainingFeatures "did not buy"
addPoints chart testFeatures "test points"
addSeparator chart trainedWeightsM
formatChart chart
