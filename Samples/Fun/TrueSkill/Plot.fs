module Plot

open TrueSkill

open Microsoft.Office.Core
open Microsoft.Office.Interop.Excel
// module OfficeCore = Microsoft.Office.Core

open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Maths

open Microsoft.FSharp.Collections

/////////////////////////////////////////////////
// Plotting
/////////////////////////////////////////////////

open Microsoft.Office.Interop.Excel

let xl = ApplicationClass()
let book = xl.Workbooks.Add() :> _Workbook
let sheet = ref null
let chart = ref null

let startNewSheet (title: string) = 
    sheet := book.Sheets.Add() :?> _Worksheet
    (!sheet).Name <- title

let startNewChart (left: float) (top: float) (width: float) (height: float) (title: string) =
    let chartObjects = (!sheet).ChartObjects() :?> ChartObjects
    chart := chartObjects.Add(left, top, width, height).Chart
    (!chart).HasTitle <- true
    (!chart).ChartTitle.Text <- title

let closeExcel () =
    xl.Quit()

let private plotBars (chart: Chart) (nPlayers: int) (playerNames: string[]) (skills: Gaussian[]): Point[] = 

    let seriesCollection = chart.SeriesCollection() :?> SeriesCollection

    chart.set_HasAxis(XlAxisType.xlCategory, XlAxisGroup.xlPrimary, false)

    // Player bars
    [| for i in 0 .. nPlayers - 1 ->
        let skill = skills.[i]
        let skillSeries = seriesCollection.NewSeries()

        skillSeries.XValues <- [| float i |]
        skillSeries.Values <- [| skill.GetMean() |]
        let stddev = skill.GetVariance() |> sqrt
        skillSeries.ChartType <- XlChartType.xlXYScatter
        skillSeries.ErrorBar(XlErrorBarDirection.xlY, XlErrorBarInclude.xlErrorBarIncludeBoth, XlErrorBarType.xlErrorBarTypeCustom, 
                                [| stddev |], [| stddev |]) |> ignore

        skillSeries.Name <- playerNames.[i]

        skillSeries.Points(1) :?> Point
    |]


let plotSkillsAndGames (maxPlayers: int) (playerNames: string[]) (games: Game[]) (skills: Gaussian[]) = 

    xl.Visible <- true

    let nPlayers = min maxPlayers playerNames.Length

    ///////////
    // Arrows
    let playerPoints = plotBars !chart nPlayers playerNames skills

    // Game arrows
    // arrow from A to B: A wins
    // green line: a draw
    let shownGames: Set<Game> ref = ref Set.empty
    for (p1, p2, outcome) as game in games do
        if p1 < nPlayers && p2 < nPlayers && not (Set.contains game !shownGames) then
            shownGames := Set.add game !shownGames
            // NB: don't ask for coordinates, if the positioning of things on the chart may change,
            // for instance, if you are still about to add new series
            let p1X, p1Y = playerPoints.[p1].Left |> float32, playerPoints.[p1].Top |> float32
            let p2X, p2Y = playerPoints.[p2].Left |> float32, playerPoints.[p2].Top |> float32
            let arrow = (!chart).Shapes.AddConnector(MsoConnectorType.msoConnectorStraight, p1X, p1Y, p2X, p2Y) 
            if outcome = 1 then
                arrow.Line.EndArrowheadStyle <- MsoArrowheadStyle.msoArrowheadTriangle
            else if outcome = -1 then
                arrow.Line.BeginArrowheadStyle <- MsoArrowheadStyle.msoArrowheadTriangle
            else if outcome = 0 then
                arrow.Line.ForeColor.SchemeColor <- 3


let plotSkills (playerNames: string[]) (skills: Gaussian[]) = 

    xl.Visible <- true

    plotBars !chart playerNames.Length playerNames skills |> ignore

    // Use layout 3 to see Excel's own linear fit
    // chart.ApplyLayout(4)

let plotCurves (maxPlayers: int) (playerNames: string[]) (skills: Gaussian[]) = 

    let bellSeriesCollection = (!chart).SeriesCollection() :?> SeriesCollection

    let nPlayers = min maxPlayers playerNames.Length

    for i = 0 to nPlayers - 1 do
        let skill = skills.[i]
        let skillSeries = bellSeriesCollection.NewSeries()
        let xvalues = [| 0.0 .. 50.0 |]
        let yvalues = [| for x in xvalues -> skill.GetLogProb(x) |> exp |]
        skillSeries.ChartType <- XlChartType.xlLine
        skillSeries.XValues <- xvalues
        skillSeries.Values <- yvalues
        skillSeries.Name <- playerNames.[i]

    // Use layout 3 to see Excel's own linear fit
    // chart.ApplyLayout(4)

