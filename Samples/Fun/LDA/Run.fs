module Run

open Interface
open Doc

open System.IO

let me = System.Reflection.Assembly.GetExecutingAssembly().Location
let dir = System.IO.Path.GetDirectoryName(me)
let path = Path.GetFullPath (dir + @"..\..\..\Test\")

let runCatsDogs() =

    let numTopics = 2
    let alpha = 0.5
    let beta = 0.1

    let docs = readExcel (path + "CatsDogs.xlsx") "A1:A3" "B1:B3"
    let vocab, docs, topics, evidence = infer 2 alpha beta docs
    printf "Model evidence: %f\n" evidence
    writeHTML path "CatsDogs" docs topics vocab

    let randomDocs = generate docs topics
    writeHTML path "CatsDogsRandom" randomDocs topics vocab

    printf "output written to %A" path

    // if on Windows, open explorer with path
    if System.Environment.OSVersion.Platform = System.PlatformID.Win32NT
    then try
          System.Diagnostics.Process.Start("explorer.exe",path) |> ignore
         with _ -> ()
   
    

// The results even for cats and dogs are rather sensitive to initialisation
MicrosoftResearch.Infer.Maths.Rand.Restart(128)

do runCatsDogs()

