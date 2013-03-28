
(*
  A simple example of inference.

  Prior assumptions:
  Either Alice or Bob dunnit, using either a gun or a pipe
  Alice dunnit 30%, Bob dunnit 70%
  Alice uses gun 3%, uses pipe 97%
  Bob uses gun 80%, uses pipe 20%
*)

module MurderMystery

open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.FSharp.Inference


[<ReflectedDefinition>]
let mystery (): bool*bool =
  let aliceDunnit = random (Bernoulli 0.30)
  let withGun = if aliceDunnit then random (Bernoulli 0.03) else random (Bernoulli 0.80)
  aliceDunnit, withGun

let (aliceDunnitPrior, withGunPrior) = inferFun2 <@ mystery @> ()

// Our prior suggests Alice didn't do it, hence Bob dunnit, most likely with a gun.
printfn "aliceDunnitPrior: %O" aliceDunnitPrior
printfn "withGunPrior: %O" withGunPrior

[<ReflectedDefinition>]
let GunFoundAtScene (gunFound:bool): bool =
  let aliceDunnit, withGun = mystery () 
  observe(withGun = gunFound)
  aliceDunnit
    
// Well, we found no gun (but a pipe) at the scene
let posterior = inferFun1 <@ GunFoundAtScene @> false 

// Given the observed pipe, the posterior suggests Alice dunnit!
printfn "aliceDunnitPosterior: %O" posterior

