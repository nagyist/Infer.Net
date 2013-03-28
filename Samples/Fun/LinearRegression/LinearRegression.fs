
(*
  Linear regression as a Bayesian model.
*)

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.Lib

open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Maths

/////////////////////////////////////////////////
// Random seed
/////////////////////////////////////////////////

// Rand.Restart(12674)

/////////////////////////////////////////////////
// Model
/////////////////////////////////////////////////

/// Generates a 'y' from an 'x' using a linear model with stochastic noise. 
/// The parameter invNoise is (1 / noise), also called precision.
[<ReflectedDefinition>]
let point (x: float) (a: float) (b: float) (invNoise: float): float * float = 
  let y = random(GaussianFromMeanAndPrecision(a * x + b, invNoise))
  x, y

[<ReflectedDefinition>]
let parameters() =
  let a = random(GaussianFromMeanAndPrecision(0.0, 1.0))
  let b = random(GaussianFromMeanAndPrecision(5.0, 0.3))
  let invNoise = random (GammaFromShapeAndScale(1.0, 1.0))
  a, b, invNoise

[<ReflectedDefinition>]
let model (data: (float * float)[]) =
  let a, b, invNoise = parameters ()
  observe (data = [| for (x, _) in data -> point x a b invNoise |])
  a, b, invNoise

/////////////////////////////////////////////////
// Data
/////////////////////////////////////////////////

let nPoints = 3

let aTrue, bTrue, invNoiseTrue = parameters()

let data = 
    [| for x in 1 .. nPoints -> point (float x) aTrue bTrue invNoiseTrue |]

/////////////////////////////////////////////////
// Plotting
/////////////////////////////////////////////////

open Microsoft.Office.Interop.Excel

let plot (data: (float*float)[]) (aTrue: float) (bTrue: float) (aInf: float) (bInf: float) = 
    let xl = ApplicationClass()
    xl.Visible <- true

    let book = xl.Workbooks.Add() :> _Workbook
    let sheet = book.Sheets.Add() :?> _Worksheet

    let chartObjects = sheet.ChartObjects() :?> ChartObjects
    let chartObject  = chartObjects.Add(100.0, 50.0, 400.0, 400.0)
    let chart = chartObject.Chart
    let series = chart.SeriesCollection() :?> SeriesCollection

    //////////
    // The data

    let dataSeries = series.NewSeries()
    let dataX, dataY = Array.unzip data
    dataSeries.XValues <- dataX
    dataSeries.Values <- dataY
    dataSeries.ChartType <- XlChartType.xlXYScatter
    dataSeries.Name <- "Data"

    //////////
    // The true line

    // No way to draw an infinite line, 
    // so drawing explicitly to the bounds of the x axis instead
    let xAxis = chart.Axes(XlAxisType.xlCategory) :?> Axis
    xAxis.MinimumScaleIsAuto <- false
    xAxis.MaximumScaleIsAuto <- false
    let xMin = xAxis.MinimumScale
    let xMax = xAxis.MaximumScale

    let trueSeries = series.NewSeries()
    trueSeries.XValues <- [| xMin; xMax |]
    trueSeries.Values <- [| aTrue * xMin + bTrue; aTrue * xMax + bTrue |]
    trueSeries.ChartType <- XlChartType.xlXYScatterLinesNoMarkers
    trueSeries.Name <- "True Line"

    //////////
    // The inferred line
    let infSeries = series.NewSeries()
    infSeries.XValues <- [| xMin; xMax |]
    infSeries.Values <- [| aInf * xMin + bInf; aInf * xMax + bInf |]
    infSeries.ChartType <- XlChartType.xlXYScatterLinesNoMarkers
    infSeries.Name <- "Inferred Line"

    // Use layout 3 to see Excel's own linear fit
    chart.ApplyLayout(4)


/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

open MicrosoftResearch.Infer.Distributions

printf "true a: %A\n" aTrue
printf "true b: %A\n" bTrue
printf "true noise (inverse): %A\n" invNoiseTrue

let aD, bD, noiseD = inferFun3 <@ model @> data
printf "inferred a: %A\n" aD
printf "inferred b: %A\n" bD
printf "inferred noise (inverse): %A\n" noiseD

let aMean = (aD :?> CanGetMean<float>).GetMean()
let bMean = (bD :?> CanGetMean<float>).GetMean()
printf "mean a: %A\n" aMean
printf "mean b: %A\n" bMean

plot data aTrue bTrue aMean bMean

