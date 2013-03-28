module Interface

open Colours
open LDA
open Doc

open MicrosoftResearch.Infer.Fun.FSharp.Syntax

open MicrosoftResearch.Infer.Distributions

open System.Collections.Generic
open System.IO
open System


/////////////////////////////////////////////////
// Input
/////////////////////////////////////////////////


let readDocs (path: string): Doc[] = 

    let files = Directory.GetFiles(path, "*.txt")

    let docs = 
        [| for file in files ->
             let name = Path.GetFileNameWithoutExtension(file)
             let text = File.ReadAllText(name)
             {title = name; text = tokenize text; topics = None}
        |]

    docs

/////////////////////////////////////////////////
// Excel input
/////////////////////////////////////////////////

open Microsoft.Office.Interop.Excel

let readExcel (filename: string) (titleRange: string) (textRange: string): Doc[] = 

    let xl = ApplicationClass()
    let book = xl.Workbooks.Open(filename)
    let sheet = book.Worksheets.[1] :?> _Worksheet

    let titleCells = sheet.get_Range(titleRange).Cells.Value2 :?> obj[,]
    let textCells= sheet.get_Range(textRange).Cells.Value2 :?> obj[,]

    let docs = 
        [| for i in 1 .. titleCells.Length ->
            let title = titleCells.[i, 1] :?> String
            let text = textCells.[i, 1] :?> String 
            {title = title; text = tokenize text; topics = None}
        |]

    book.Close()
    xl.Quit()

    docs


/////////////////////////////////////////////////
// HTML output
/////////////////////////////////////////////////


(*
    Need to pass in vocabulary as argument because in randomly generated documents some of the
    words may be missing, so we cannot restore the full vocabulary from those documents.
*)
let writeHTML (path: string) (title: string) (docs: Doc[]) (topics: Topic[]) (vocab: Vocabulary): unit = 

    let outputDir = "output" + title

    let dirPath = path + outputDir + @"\"

    let topicDistForWord (word: int): float[] = 
            [| for topic in topics -> topic.GetMean().[word]|]

    (*
        Given a topic distribution of a word, return a measure of the discriminative power of the word.
        For instance, "and" has low discriminative power, as it is equally likely to belong to any topic,
        whereas "Infer.NET" might have high discriminative power.

        The metric below is just one possible heuristic. A better way would be to use the tf*idf metric.
    *)
    let discriminativePower (topicDist: float[]): float = 
        let sortedTopics = Array.sort topicDist
        let power = sortedTopics.[topicDist.Length - 1] / sortedTopics.[topicDist.Length - 2]
        power

    let writeIndexFile () =
        use sw = new StreamWriter(path + "results" +  title + ".html")

        let html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Frameset//EN\" \"http://www.w3.org/TR/html4/frameset.dtd\">
            <html>
            <head>
            <title>LDA Results</title>
            </head>

            <frameset cols=\"550,*\" frameborder=\"0\" border=\"0\" framespacing=\"0\">
              <frameset rows=\"250px,*\" frameborder=\"0\" border=\"0\" framespacing=\"0\">
                <frame name=\"topics\" src=\"" + outputDir + "/topics.html\" marginheight=\"0\" marginwidth=\"0\" scrolling=\"auto\" noresize>
                <frame name=\"docs\" src=\"" + outputDir + "/docs.html\" marginheight=\"0\" marginwidth=\"0\" scrolling=\"auto\" noresize>
              </frameset>
	            <frame name=\"content\" marginheight=\"0\" marginwidth=\"0\" scrolling=\"auto\" noresize>

            <noframes>
            <p>Your browser doesn't support frames.</p>
            </noframes>

            </frameset>
            </html>"

        sw.WriteLine(html)

    let colourBox (weight: float) (topicDist: float[]): string = 
        let totalWidth = 100.0 * weight // in pixel
        let totalDist = Array.sum topicDist
        let cells = 
            [| for i in 0 .. topicDist.Length - 1 -> 
                let width = (topicDist.[i] / totalDist) * totalWidth // in pixel
                if width >= 1.0 then String.Format("<td width=\"{0}px\" bgcolor=\"{1}\"></td>", width, colours.[i])
                else "" 
             |]
        String.Format("<table border=\"0\" height=\"10px\" style=\"display:inline\" ><tr>{0}</tr></table>", 
                        String.concat "\n" cells) 

    let newItem = "</br>"

    let docName (i: int): string = "Document" + (string i) 
    let topicName (i: int): string = "Topic" + (string i)

    let link (href: string) (colour: string option) (text: string): string = 
        match colour with
        | Some colour ->
            String.Format("<a href=\"{0}\" style=\"color: {1}\" target=content>{2}</a>", href, colour, text)
        | None ->
            String.Format("<a href=\"{0}\" target=content>{1}</a>", href, text)

    let colour (colour: string) (text: string): string = 
        String.Format("<span colour={0}>{1}</span>", colour, text)

    let writeTopicList (nTopics: int): unit =
    
        use sw = new StreamWriter(dirPath + "topics.html")

        for i in 0 .. nTopics - 1 do
            sw.WriteLine(newItem)
            let name = topicName i
            sw.WriteLine(name |> link (name + ".html") (Some colours.[i]))

    let writeDocList (docs: Doc[]): unit =

        use sw =  new StreamWriter(dirPath + "docs.html")

        for i in 0 .. docs.Length - 1 do
            let doc = docs.[i]
            sw.WriteLine(newItem)
            sw.WriteLine(doc.title |> link (docName i + ".html") None)
            match doc.topics with
            | Some topics -> sw.WriteLine(colourBox 1.0 (topics.GetMean().ToArray()))
            | None -> ()

    let writeTopic (topic: Topic) (i: int): unit = 

        let numWordsToPrint = min 20 topic.Dimension

        let name = topicName i

        use sw =  new StreamWriter(dirPath + name + ".html")

        sw.WriteLine("Top {0} words in topic {1}:", numWordsToPrint, name)
        sw.WriteLine(newItem)
        sw.WriteLine("showing (relative) contribution of each word to each topic, absolute word frequency not shown")

        let pc = topic.PseudoCount.ToArray()
        let wordIndices = [| for j in 0 .. pc.Length - 1 -> j |]

        System.Array.Sort(pc, wordIndices)

        let highlightedWords = 
            [| for j in 1 .. numWordsToPrint ->
                let id = wordIndices.[wordIndices.Length - j]
                let word = id2word vocab id
                let topicDists = topicDistForWord id
                word + (colourBox 1.0 topicDists) 
            |]

        for word in highlightedWords do
            sw.WriteLine(newItem)
            sw.WriteLine(word)

    let writeDoc (doc: Doc) (i: int): unit =

        use sw =  new StreamWriter(dirPath + docName i + ".html")

        let words = doc.text |> wordIds vocab

        let maxPower = Array.max [| for word in words -> discriminativePower (topicDistForWord word)|]

        let highlightedWords = 
            [| for token in doc.text -> 
                match token with
                | Word word -> 
                    let topicDists = topicDistForWord (word2id vocab word)
                    word + (colourBox (discriminativePower topicDists / maxPower) topicDists) 
                | NotWord t -> t |]

        let title = "<h2>" + doc.title + "</h2>"

        title + String.concat " " highlightedWords |> sw.WriteLine

    Directory.CreateDirectory dirPath |> ignore

    writeIndexFile ()

    writeTopicList topics.Length
    writeDocList docs

    for i in 0 .. topics.Length - 1 do
        writeTopic topics.[i] i

    for i in 0 .. docs.Length - 1 do
        writeDoc docs.[i] i
