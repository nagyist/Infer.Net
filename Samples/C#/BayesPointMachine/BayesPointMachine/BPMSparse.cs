// (C) Copyright 2008 Microsoft Research Cambridge
using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions.Kernels;
using MicrosoftResearch.Infer.Utils;

namespace BayesPointMachine
{
	/// <summary>
	/// Sparse Bayes point machine - model variables for training
	/// </summary>
	public class BPMSparseVarsForTrain
	{
		public VariableArray<double>[] w;
		public VariableArray<Gaussian>[] wInit;
		public BPMDataVars[] dataVars;  // dataVars[k]: variables for kth component
		public InferenceEngine ie;
	}

	/// <summary>
	/// Sparse Bayes point machine - model variables for testing
	/// </summary>
	public class BPMSparseVarsForTest
	{
		public VariableArray<double>[] w;
		public VariableArray<Gaussian>[] wPrior;
		public BPMDataVars dataVars;
		public InferenceEngine ie;
		public VariableArray<int> y;
	}

	/// <summary>
	/// Sparse Bayes point machine
	/// </summary>
	public class BPMSparse
	{
		BPMSparseVarsForTrain trainModel;
		BPMSparseVarsForTest testModel;
		int nClass, nFeatures;
		Range feature;

		/// <summary>
		/// Engine for training
		/// </summary>
		public InferenceEngine TrainingEngine { get { return trainModel.ie; } }
		/// <summary>
		/// Engine for testing
		/// </summary>
		public InferenceEngine TestEngine { get { return testModel.ie; } }

		/// <summary>
		/// Noise precision
		/// </summary>
		public double NoisePrec { get; private set; }

		/// <summary>
		/// Constructs a Sparse Bayes point machine instance from number of components and number of features
		/// </summary>
		/// <param name="nClass">Number of components (classes)</param>
		/// <param name="nFeatures">Number of features</param>
		/// <param name="noisePrec">noisePrec</param>
		public BPMSparse(int nClass, int nFeatures, double noisePrec)
		{
			this.nClass = nClass;
			this.nFeatures = nFeatures;
			NoisePrec = noisePrec;
			feature = new Range(nFeatures).Named("feature");
			trainModel = SpecifyTrainModel("_train");
			testModel = SpecifyTestModel("_test");
		}

		/// <summary>
		/// Specifies the training model
		/// </summary>
		/// <param name="s">The name of the training model</param>
		/// <returns>A <see cref="BPMSparseVarsForTrain"/> instance</returns>
		private BPMSparseVarsForTrain SpecifyTrainModel(string s)
		{
			// Place to maintain variables for each component
			BPMDataVars[] dataVars= new BPMDataVars[nClass];
			// The weight vector for each component
			VariableArray<double>[] w = new VariableArray<double>[nClass];
			// The prior weight distributions for each component
			VariableArray<Gaussian>[] wInit = new VariableArray<Gaussian>[nClass];
			for (int c = 0; c < nClass; c++) {
				// Weight variables are an array over feature range
				w[c] = Variable.Array<double>(feature).Named("w_" + c + s);
				// The prior weight distributions will be set by the calling program
				wInit[c] = Variable.Array<Gaussian>(feature).Named("wInit_" + c + s);
				// The weight for this component and feature
				w[c][feature] = Variable<double>.Random(wInit[c][feature]);
			}
			// Loop over the components
			for (int c = 0; c < nClass; c++) {
				Variable<int> nItem = Variable.New<int>().Named("nItem_" + c + s);
				Range item = new Range(nItem).Named("item_" + c + s);
				// Array of feature counts per item
				VariableArray<int> xValueCount = Variable.Array<int>(item).Named("xCount_" + c + s);
				// Range over features - size is based on variable feature count
				Range itemFeature = new Range(xValueCount[item]).Named("itemFeature_" + c + s);
				// Jagged array of values - each item is an array of data values whose indices
				// are given by the corresping xIndices[item]
				VariableArray<VariableArray<double>, double[][]> xValues = Variable.Array(Variable.Array<double>(itemFeature), item).Named("xValues_" + c + s);
				// Jagged array of indices for the items
				VariableArray<VariableArray<int>, int[][]> xIndices = Variable.Array(Variable.Array<int>(itemFeature), item).Named("xIndices_" + c + s);
				// Loop over the data items
				using (Variable.ForEach(item)) {
					// The score for this item across all components
					Variable<double>[] score = BPMUtils.ComputeClassScores(w, xValues[item], xIndices[item], itemFeature, NoisePrec);
					// The constraint imposed by the observed component
					BPMUtils.ConstrainArgMax(c, score);
				}
				// Store the data information
				dataVars[c] = new BPMDataVars(nItem, item, xIndices, xValueCount, xValues);
			}
			// Store the variables 
			BPMSparseVarsForTrain bpmVar = new BPMSparseVarsForTrain();
			bpmVar.dataVars = dataVars;
			bpmVar.ie = new InferenceEngine();
			bpmVar.ie.ModelName = "BPMSparse_train";
			bpmVar.w = w;
			bpmVar.wInit = wInit;
			return bpmVar;
		}

		/// <summary>
		/// Specifies the test model
		/// </summary>
		/// <param name="s">The name of the test model</param>
		/// <returns>A <see cref="BPMSparseVarsForTest"/> instance</returns>
		private BPMSparseVarsForTest SpecifyTestModel(string s)
		{
			// The weight vector for each component
			VariableArray<double>[] w = new VariableArray<double>[nClass];
			// The prior distribution for weight vector for each component. When
			// <see cref="Test"/> is called, this is set to the posterior weight
			// distributions from <see cref="Train"/>
			VariableArray<Gaussian>[] wPrior = new VariableArray<Gaussian>[nClass];
			for (int c = 0; c < nClass; c++) {
				// Weight variables are an array over feature range
				wPrior[c] = Variable.Array<Gaussian>(feature);
				w[c] = Variable.Array<double>(feature).Named("w" + c + s);
				// The weight for this component and feature
				w[c][feature] = Variable<double>.Random(wPrior[c][feature]);
			}
			// The number of items
			Variable<int> nItem = Variable.New<int>().Named("nItem_" +s);
			// Range over the number of items
			Range item = new Range(nItem).Named("item_" +s);
			// Array of feature counts per item
			VariableArray<int> xValueCount = Variable.Array<int>(item).Named("xCount_" + s);
			// Range over features - size is based on variable feature count
			Range itemFeature = new Range(xValueCount[item]).Named("itemFeature" + s);
			// Jagged array of values - each item is an array of data values whose indices
			// are given by the corresping xIndices[item]
			VariableArray<VariableArray<double>, double[][]> xValues = Variable.Array(Variable.Array<double>(itemFeature), item).Named("xValues" + s);
			// Jagged array of indices for the items
			VariableArray<VariableArray<int>, int[][]> xIndices = Variable.Array(Variable.Array<int>(itemFeature), item).Named("xIndices" + s);
			VariableArray<int> ytest = Variable.Array<int>(item).Named("ytest" + s);
			using (Variable.ForEach(item))
			{
				// The score for this item across all components
				Variable<double>[] score = BPMUtils.ComputeClassScores(w, xValues[item], xIndices[item], itemFeature, NoisePrec);
				ytest[item] = Variable.Discrete(Vector.Constant(nClass, 1.0 / nClass));
				// The constraints on the output variable
				BPMUtils.ConstrainMaximum(ytest[item], score, nClass);
			}
			// Store the variables 
			BPMSparseVarsForTest bpmVar = new BPMSparseVarsForTest();
			bpmVar.ie = new InferenceEngine();
			bpmVar.ie.ModelName = "BPMSparse_test";
			bpmVar.dataVars = new BPMDataVars(nItem, item, xIndices, xValueCount, xValues); ;
			bpmVar.y = ytest;
			bpmVar.w = w;
			bpmVar.wPrior = wPrior;
			return bpmVar;
		}

		/// <summary>
		/// Test this sparse BPM
		/// </summary>
		/// <param name="xIndicesData">Jagged array of data indices; xIndicesData[t]: feature indices for t^{th} datapoint</param>
		/// <param name="xValuesData">Jagged array of data values; xValuesData[t]: feature values for t^{th} datapoint </param>
		/// <returns>posterior distribution over the classes for each datapoint</returns>
		public Discrete[] Test(int[][] xIndicesData, double[][] xValuesData)
		{
			// Set the observed values
			for (int c = 0; c < nClass; c++) {
				testModel.wPrior[c].ObservedValue = trainModel.wInit[c].ObservedValue;
			}
			testModel.dataVars.SetObservedValues(xIndicesData, xValuesData);
			// Infer the outputs
			return Distribution.ToArray<Discrete[]>(testModel.ie.Infer(testModel.y));
		}

		/// <summary>
		/// Train this sparse BPM
		/// </summary>
		/// <param name="xIndicesData">Jagged array of indices for all classes; xIndicesData[k][t]: feature indices for t^{th} datapoint of k^{th} class</param>
		/// <param name="xValuesData">Jagged array of data for all classes; xValuesData[k][t]: feature values for t^{th} datapoint of k^{th} class</param>
		/// <returns>posterior distribution over weights for each class</returns>
		public Gaussian[][] Train(int[][][] xIndicesData, double[][][] xValuesData)
		{
			Gaussian[] wprior0 = new Gaussian[nFeatures];
			Gaussian[] wprior = new Gaussian[nFeatures];
			for (int f = 0; f < nFeatures; f++)
			{
				wprior0[f] = Gaussian.PointMass(0.0);
				wprior[f] = Gaussian.FromMeanAndPrecision(0.0, 1.0);
			}
			for (int c = 0; c < nClass; c++)
				trainModel.wInit[c].ObservedValue = (c==0) ? wprior0 : wprior;

			return InferW(xIndicesData, xValuesData);
		}

		/// <summary>
		/// Train this sparse BPM incrementally
		/// </summary>
		/// <param name="xIndicesData">Jagged array of indices for all classes; xIndicesData[k][t]: feature indices for t^{th} datapoint of k^{th} class</param>
		/// <param name="xValuesData">Jagged array of data for all classes; xValuesData[k][t]: feature values for t^{th} datapoint of k^{th} class</param>
		/// <returns>posterior distribution over weights for each class</returns>
		public Gaussian[][] TrainIncremental(int[][][] xIndicesData, double[][][] xValuesData)
		{
			return ((trainModel.wInit[0].IsObserved) ? InferW(xIndicesData, xValuesData) : Train(xIndicesData, xValuesData));
		}

		private Gaussian[][] InferW(int[][][] xIndicesData, double[][][] xValuesData)
		{
			//set Observed values
			for (int c = 0; c < nClass; c++) {
				trainModel.dataVars[c].SetObservedValues(xIndicesData[c], xValuesData[c]);
			}
			Gaussian[][] wInferred = new Gaussian[nClass][];
			for (int c = 0; c < nClass; c++) {
				wInferred[c] = (trainModel.ie).Infer<Gaussian[]>(trainModel.w[c]);
				trainModel.wInit[c].ObservedValue = wInferred[c];
			}
			return wInferred;
		}
	}
}
