// (C) Copyright 2008 Microsoft Research Cambridge
using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions.Kernels;
using MicrosoftResearch.Infer.Utils;
using GaussianArray=MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;
namespace BayesPointMachine
{
	class BPMUtils
	{
		/// <summary>
		/// Compute class scores for sparse BPM
		/// </summary>
		/// <param name="w">Weight array per component</param>
		/// <param name="xValues">Vector  of values</param>
		/// <param name="noisePrec">Noise precision</param>
		/// <returns>score for each class</returns>
		public static Variable<double>[] ComputeClassScores(Variable<Vector>[] w, Variable<Vector> xValues, double noisePrec)
		{
			int nClass = w.Length;
			Variable<double>[] score = new Variable<double>[nClass];
			Variable<double>[] scorePlusNoise = new Variable<double>[nClass];
			for (int c = 0; c < nClass; c++)
			{
				score[c] = Variable.InnerProduct(w[c], xValues);
				scorePlusNoise[c] = Variable.GaussianFromMeanAndPrecision(score[c], noisePrec);
			}
			return scorePlusNoise;
		}

		/// <summary>
		/// Compute class scores for sparse BPM
		/// </summary>
		/// <param name="w">Weight array per class</param>
		/// <param name="xValues">Array of values</param>
		/// <param name="xIndices">Array of indices</param>
		/// <param name="itemFeature">Feature range</param>
		/// <param name="noisePrec">Noise precision</param>
		/// <returns></returns>
		public static Variable<double>[] ComputeClassScores(
			VariableArray<double>[] w, VariableArray<double> xValues,
			VariableArray<int> xIndices, Range itemFeature, double noisePrec)
		{
			int nClass = w.Length;
			Variable<double>[] score = new Variable<double>[nClass];
			Variable<double>[] scorePlusNoise = new Variable<double>[nClass];
			for (int c = 0; c < nClass; c++)
			{
				VariableArray<double> wSparse = Variable.Subarray<double>(w[c], xIndices);
				VariableArray<double> product = Variable.Array<double>(itemFeature);
				product[itemFeature] = xValues[itemFeature] * wSparse[itemFeature];
				score[c] = Variable.Sum(product);
				scorePlusNoise[c] = Variable.GaussianFromMeanAndPrecision(score[c], noisePrec);
			}
			return scorePlusNoise;
		}

		/// <summary>
		/// Builds a multicomponent switch for the specified integer variable
		/// which builds a set of <see cref="ConstrainArgMax"/> constraints based
		/// on the value of the variable
		/// </summary>
		/// <param name="ytrain">The specified integer variable</param>
		/// <param name="score">The vector of score variables</param>
		/// <param name="nClass">The number of components</param>
		public static void ConstrainMaximum(Variable<int> ytrain, Variable<double>[] score, int nClass)
		{
			for (int c = 0; c < nClass; c++)
			{
				using (Variable.Case(ytrain, c))
				{
					ConstrainArgMax(c, score);
				}
			}
		}

		/// <summary>
		/// Constrain the score for the specified component to be bigger than
		/// all the scores at the other components
		/// </summary>
		/// <param name="argmax">The sprecified index</param>
		/// <param name="score">The vector of score variables</param>
		public static void ConstrainArgMax(int argmax, Variable<double>[] score)
		{
			for (int c = 0; c < score.Length; c++)
			{
				if (c != argmax)
					Variable.ConstrainPositive(score[argmax] - score[c]);
			}
		}
	}
}
