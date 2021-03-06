﻿using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MicrosoftResearch.Infer.Tutorials
{
	[Example("Applications", "The Rats example from BUGS")]
	public class BugsRats
	{
		public void Run()
		{
			Rand.Restart(12347);

			// The model
			int N = RatsHeightData.GetLength(0);
			int T = RatsHeightData.GetLength(1);
			Range r = new Range(N).Named("N");
			Range w = new Range(T).Named("T");

			Variable<double> alphaC = Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("alphaC");
			Variable<double> alphaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("alphaTau");
			VariableArray<double> alpha = Variable.Array<double>(r).Named("alpha");
			alpha[r] = Variable.GaussianFromMeanAndPrecision(alphaC, alphaTau).ForEach(r);

			Variable<double> betaC = Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("betaC");
			Variable<double> betaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("betaTau");
			VariableArray<double> beta = Variable.Array<double>(r).Named("beta");
			beta[r] = Variable.GaussianFromMeanAndPrecision(betaC, betaTau).ForEach(r);

			Variable<double> tauC = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("tauC");
			VariableArray<double> x = Variable.Observed<double>(RatsXData, w).Named("x");
			Variable<double> xbar = Variable.Sum(x)/T;
			VariableArray2D<double> y = Variable.Observed<double>(RatsHeightData, r, w).Named("y");
			y[r, w] = Variable.GaussianFromMeanAndPrecision(alpha[r] + (beta[r] * (x[w]-xbar)), tauC);
			Variable<double> alpha0 = (alphaC - betaC * xbar).Named("alpha0");

			// Initialise with the mean of the prior (needed for Gibbs to converge quickly)
			alphaC.InitialiseTo(Gaussian.PointMass(0.0));
			tauC.InitialiseTo(Gamma.PointMass(1.0));
			alphaTau.InitialiseTo(Gamma.PointMass(1.0));
			betaTau.InitialiseTo(Gamma.PointMass(1.0));

			// Inference engine
			InferenceEngine ie = new InferenceEngine();
			if (!(ie.Algorithm is ExpectationPropagation))
			{
				Gaussian betaCMarg = ie.Infer<Gaussian>(betaC);
				Gaussian alpha0Marg = ie.Infer<Gaussian>(alpha0);
				Gamma tauCMarg = ie.Infer<Gamma>(tauC);

				// Inference
				Console.WriteLine("alpha0 = {0}[sd={1}]", alpha0Marg, Math.Sqrt(alpha0Marg.GetVariance()).ToString("g4"));
				Console.WriteLine("betaC = {0}[sd={1}]", betaCMarg, Math.Sqrt(betaCMarg.GetVariance()).ToString("g4"));
				Console.WriteLine("tauC = {0}", tauCMarg);
			}
			else
				Console.WriteLine("This example does not run with Expectation Propagation");
		}

		// Height data
		double[,] RatsHeightData = new double[,]
			{{151, 199, 246, 283, 320},
			 {145, 199, 249, 293, 354},
			 {147, 214, 263, 312, 328},
			 {155, 200, 237, 272, 297},
			 {135, 188, 230, 280, 323},
			 {159, 210, 252, 298, 331},
			 {141, 189, 231, 275, 305},
			 {159, 201, 248, 297, 338},
			 {177, 236, 285, 350, 376},
			 {134, 182, 220, 260, 296},
			 {160, 208, 261, 313, 352},
			 {143, 188, 220, 273, 314},
			 {154, 200, 244, 289, 325},
			 {171, 221, 270, 326, 358},
			 {163, 216, 242, 281, 312},
			 {160, 207, 248, 288, 324},
			 {142, 187, 234, 280, 316},
			 {156, 203, 243, 283, 317},
			 {157, 212, 259, 307, 336},
			 {152, 203, 246, 286, 321},
			 {154, 205, 253, 298, 334},
			 {139, 190, 225, 267, 302},
			 {146, 191, 229, 272, 302},
			 {157, 211, 250, 285, 323},
			 {132, 185, 237, 286, 331},
			 {160, 207, 257, 303, 345},
			 {169, 216, 261, 295, 333},
			 {157, 205, 248, 289, 316},
			 {137, 180, 219, 258, 291},
			 {153, 200, 244, 286, 324}
			};

		// x data
		double[] RatsXData = { 8.0, 15.0, 22.0, 29.0, 36.0 };
	}
}
