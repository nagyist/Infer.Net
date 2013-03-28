module Doc

open MicrosoftResearch.Infer.Distributions

open System
open System.Collections.Generic

open LDA



/////////////////////////////////////////////////
// Tokenizing text
/////////////////////////////////////////////////

let private isNonWord c = 
    (System.Char.IsWhiteSpace c || System.Char.IsPunctuation c) && not (System.Char.IsSymbol c) && not (c = '#')

let private stopWords = 
    [| "of"; "and"; "the"; "a"; "an"; "in"; "to"; "for"; "we"; "on"; "from"; "is"; "new";
        "that"; "with"; "can"; "will"; "as"; "such"; "s"; "we"; "are"; "you"; "these"; "so"; "this";
        "which"; "also"; "use"; "used"; "have"; "up"; "or"; "be"; "it"; "i"; "our"; "used"; "by"; "being";
        "what"; "do" ; "as"; "they"; "way"; "how"; "your"; "now"; "more"; "than"; "their" |]

let private chars2token (cs: char list) = 
    let word = String.Concat cs
    if Array.exists (fun w -> w = word.ToLower()) stopWords then
        NotWord word
    else
        Word word

// may generate empty words, the caller must filter
let rec private tokenize_: char list -> char list -> Token list = fun word text ->
    match word, text with
    | word, c :: cs when isNonWord c -> 
        chars2token word :: NotWord (string c) :: tokenize_ [] cs
    | word, c :: cs -> 
        tokenize_ (word @ [c]) cs
    | word, [] -> [chars2token word]

let tokenize (text: string): Token[] =
    text.ToCharArray() |> List.ofArray |> tokenize_ [] |> List.filter (function | Word "" -> false | _ -> true) |> Array.ofList 
    // text.Split( [| ' '; '\n'; '\t'; '\r'; '.'; ','; ';'; '\"'|], StringSplitOptions.RemoveEmptyEntries )   


/////////////////////////////////////////////////
// Vocabulary
// A mapping between words and integers
/////////////////////////////////////////////////

type Vocabulary = Dictionary<string, int> * Dictionary<int, string>

/// Words that differ in case map to the same id.
let word2id (v: Vocabulary) (word: string): int =
    let ivocab, vocab = v
    let word = word.ToLower()
    if not (ivocab.ContainsKey(word)) then
        ivocab.[word] <- vocab.Count
        vocab.[vocab.Count] <- word
    ivocab.[word]

let id2word (v: Vocabulary) (id: int): string = 
    let ivocab, vocab = v in vocab.[id] 

let wordIds (v: Vocabulary) (tokens: Token[]): int[] =
    
    let getId = function
        | Word "" -> None
        | NotWord _ -> None
        | Word word -> Some (word2id v word)

    Array.choose getId tokens

let encode (docs: Doc[]): int[][] * Vocabulary =
    let vocab = new Dictionary<string, int>(), new Dictionary<int, string>()
    let words = [| for doc in docs -> wordIds vocab doc.text |]
    let (v, iv) = vocab

    // debug output
    use sw = new System.IO.StreamWriter("words.out")
    for word in v.Keys do
        fprintf sw "%s\n" word

    words, vocab


let vocabularySize (v: Vocabulary): int =
    let ivocab, vocab = v
    vocab.Count

/////////////////////////////////////////////////
// Inference
/////////////////////////////////////////////////

let infer (numTopics: int) (alpha: float) (beta: float) (docs: Doc[]): Vocabulary * Doc[] * Topic[] * float = 

    let encodedDocs, vocab = encode docs
    let sizeVocab = vocabularySize vocab

    Console.WriteLine("************************************")
    Console.WriteLine("Vocabulary size = " + string sizeVocab)
    Console.WriteLine("Number of documents = " + string docs.Length)
    Console.WriteLine("Number of topics = " + string numTopics)
    Console.WriteLine("alpha = " + string alpha)
    Console.WriteLine("beta = " + string beta)
    Console.WriteLine("************************************")

    let postTheta, postPhi, evidence = LDA.infer sizeVocab numTopics alpha beta encodedDocs

    let docs' = [|for i in 0 .. postTheta.Length - 1 -> {docs.[i] with topics = Some postTheta.[i]}|]

    vocab, docs', postPhi, evidence


/////////////////////////////////////////////////
// Generation
/////////////////////////////////////////////////

let generate (docs: Doc[]) (topics: Topic[]): Doc[] = 
    let theta = [| for doc in docs -> match doc.topics with Some t -> t.GetMean() | None -> failwith "Infer topics first" |]
    let phi = [| for t in topics -> t.GetMean() |]
    let encodedDocs, vocab = encode docs

    let randomWords = LDA.generateWords(theta, phi, encodedDocs)

    let docs' = 
        [| for i in 0 .. docs.Length - 1 ->
            {docs.[i] with text = Array.map (function id -> Word (id2word vocab id)) randomWords.[i] }
        |]

    docs'
