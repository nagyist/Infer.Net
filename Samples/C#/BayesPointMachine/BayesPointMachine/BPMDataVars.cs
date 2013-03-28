// (C) Copyright 2008 Microsoft Research Cambridge
using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Collections;
using GaussianArray=MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;

namespace BayesPointMachine
{
	/// <summary>
	/// Class which maintains all the data variables for a sparse (and shared) BPM
	/// </summary>
	public class BPMDataVars
	{
		public VariableArray<VariableArray<double>, double[][]> xValues;
		public VariableArray<VariableArray<int>, int[][]> xIndices;
		public Variable<int> nItem;
		public Range item, itemFeature;
		public VariableArray<int> xValueCount;

		public BPMDataVars()
		{
		}

		public BPMDataVars(
			Variable<int> nUser, Range user, 
			VariableArray<VariableArray<int>, int[][]> xIndices,
			VariableArray<int> xValueCount,
			VariableArray<VariableArray<double>, double[][]> xValues)
		{
			this.nItem = nUser;
			this.item = user;
			this.xValueCount = xValueCount;
			this.xValues = xValues;
			this.xIndices = xIndices;
		}

		public void SetObservedValues(int[][] xIndicesData, double[][] xValuesData)
		{
			nItem.ObservedValue = xIndicesData.Length;
			xValues.ObservedValue = xValuesData;
			xIndices.ObservedValue = xIndicesData;

			int[] xValueCountData = new int[xIndicesData.Length];
			for (int i = 0; i < xIndicesData.Length; i++)
			{
				xValueCountData[i] = xIndicesData[i].Length;
			}
			xValueCount.ObservedValue = xValueCountData;
		}
	}

}

