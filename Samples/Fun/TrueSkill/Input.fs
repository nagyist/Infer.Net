module Input

open TrueSkill

open System.Text.RegularExpressions

/////////////////////////////////////////////////
// Input
/////////////////////////////////////////////////


let read_lines filename line = 
    use f = System.IO.File.OpenRead(filename)
    use s = new System.IO.StreamReader(f)
    seq { while not s.EndOfStream do 
            yield line <| s.ReadLine()
    } |> Seq.toArray

let idnum s = (s |> int32) - 1

let read_player =
    let player_re = new Regex("(\d+),(\w*)\s*,(\w*)\s*,\d+,\d+,\w*\s*")
    fun line ->
        let m = player_re.Match(line)
        let id = m.Groups.[1].Value |> idnum
        let last = m.Groups.[2].Value
        let first = m.Groups.[3].Value
        id,first,last

let read_game =
    let game_re = new Regex("(\d+),(\d+),(\d+),\d+")
    fun line ->
        let m = game_re.Match(line)
        let id = m.Groups.[1].Value |> idnum
        let player = m.Groups.[2].Value |> idnum
        let score = m.Groups.[3].Value |> int32
        id,player,score

/// Return the list of player names and the list of games
let load (playersFile: string) (gamesFile: string): string[] * Game[] =
    let raw_players = read_lines playersFile read_player
    let raw_games = read_lines gamesFile read_game
    
    let players = 
        [|for (_, first, last) in raw_players -> first + " " + last|]

    let games = 
        [|for i in 0 .. 2 .. raw_games.Length-1 -> 
            let (_,p1,s1) = raw_games.[i]
            let (_,p2,s2) = raw_games.[i+1]
            if s1 > s2   then (p1, p2, 1)     // player 1 beats player 2
            elif s2 > s1 then (p1, p2, -1)    // player 1 beaten by player 2
                         else (p1, p2, 0)|]   // the players draw
    players, games
