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

namespace BayesPointMachine
{
	/// <summary>
	/// Bayes point machine - model variables for training
	/// </summary>
	public class BPMVarsModelForTrain
	{
		public InferenceEngine ie;
		public Variable<int>[] nItems;
		public VariableArray<Vector>[] xValues;
		public Model model;
	}

	/// <summary>
	/// Bayes point machine - model variables for testing
	/// </summary>
	public class BPMVarsModelForTest
	{
		public InferenceEngine ie;
		public Variable<int> nItems;
		public VariableArray<Vector> xValues;
		public Model model;
		public VariableArray<int> y;
	}

	/// <summary>
	/// Bayes point machine using shared variables
	/// </summary>
	public class BPM_Shared
	{
		BPMVarsModelForTrain trainModel;
		BPMVarsModelForTest testModel;
		int nClass, nFeatures;

		SharedVariable<Vector>[] w;
		int trainChunkSize, testChunkSize;
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
		/// Constructs a multi-component Bayes Point Machine using shared variables for chunking data
		/// </summary>
		/// <param name="nClass">Number of components (classes)</param>
		/// <param name="nFeatures">Number of features</param>
		/// <param name="noisePrec">Noise precision</param>
		/// <param name="trainChunkSize">Chunk size for training</param>
		/// <param name="testChunkSize">Chunk size for testing</param>
		public BPM_Shared(int nClass, int nFeatures, double noisePrec, int trainChunkSize, int testChunkSize)
		{
			this.nClass = nClass;
			this.nFeatures = nFeatures;
			this.trainChunkSize = trainChunkSize;
			this.testChunkSize = testChunkSize;
			NoisePrec = noisePrec;

			feature = new Range(nFeatures).Named("feature");

			// The set of weight vectors (one for each component) are shared between all data chunks
			w = new SharedVariable<Vector>[nClass];
			VectorGaussian wPrior0 = VectorGaussian.PointMass(Vector.Zero(nFeatures));
			VectorGaussian wPrior = VectorGaussian.FromMeanAndPrecision(Vector.Zero(nFeatures), PositiveDefiniteMatrix.Identity(nFeatures));
			for (int c = 0; c < nClass; c++)
			{
				w[c] = (c == 0)
					? SharedVariable<Vector>.Random(VectorGaussian.PointMass(Vector.Zero(nFeatures))).Named("w_" + c)
					: SharedVariable<Vector>.Random(wPrior).Named("w_" + c);
			}
			trainModel = SpecifyTrainModel("_train", trainChunkSize);
			testModel = SpecifyTestModel("_test", testChunkSize);
		}

		/// <summary>
		/// Specify the training model
		/// </summary>
		/// <param name="s">The name of the training model</param>
		/// <param name="nChunks">The number of chunks</param>
		/// <returns>A <see cref="BPMVarsModelForTrain"/> instance</returns>
		private BPMVarsModelForTrain SpecifyTrainModel(string s, int nChunks)
		{
			// An array of feature vectors - their observed values will be
			// set by the calling program
			VariableArray<Vector>[] xValues = new VariableArray<Vector>[nClass];
			// The number of items for each component
			Variable<int>[] nItem = new Variable<int>[nClass];
			// Ranges over the items for each component
			Range[] item = new Range[nClass];

			// The model identifier for the shared variables
			Model model = new Model(nChunks).Named("model" + s);
			// The weight vector within a submodel
			Variable<Vector>[] wModel = new Variable<Vector>[nClass];
			for (int c = 0; c < nClass; c++)
			{
				// Get a copy of the shared weight vector variable for the submodel
				wModel[c] = w[c].GetCopyFor(model).Named("wModel_" + c + s);
			}

			// Loop over the components
			for (int c = 0; c < nClass; c++)
			{
				// The number of items for each component - set by the calling program
				nItem[c] = Variable.New<int>().Named("nItem_" + c + s);
				// The item range for each component
				item[c] = new Range(nItem[c]).Named("item_" + c + s);
				// The items for each component - set by the calling program
				xValues[c] = Variable.Array<Vector>(item[c]);
				using (Variable.ForEach(item[c]))
				{
					// The score for this item across all components
					Variable<double>[] score = BPMUtils.ComputeClassScores(wModel, xValues[c][item[c]], NoisePrec);
					// The constraint imposed by the observed component
					BPMUtils.ConstrainArgMax(c, score);
				}
			}
			// Store the variables 
			BPMVarsModelForTrain bpmVar = new BPMVarsModelForTrain();
			bpmVar.ie = new InferenceEngine();
			bpmVar.xValues = xValues;
			bpmVar.nItems = nItem;
			bpmVar.model = model;
			return bpmVar;
		}

		/// <summary>
		/// Specify the training model
		/// </summary>
		/// <param name="s">The name of the test model</param>
		/// <param name="nChunks">The number of chunks</param>
		/// <returns>A <see cref="BPMVarsModelForTest"/> instance</returns>
		private BPMVarsModelForTest SpecifyTestModel(string s, int nChunks)
		{
			// The number of test items - this will be set by the calling program
			Variable<int> nItem = Variable.New<int>().Named("nItem" + s);
			// A range over the items
			Range item = new Range(nItem).Named("item" + s);
			// An array of feature vectors - their observed values will be
			// set by the calling program
			VariableArray<Vector> xValues = Variable.Array<Vector>(item);
			// The model identifier for the shared variables
			Model model = new Model(nChunks).Named("model" + s);
			// The weight vector for each submodel
			Variable<Vector>[] wModel = new Variable<Vector>[nClass];
			for (int c = 0; c < nClass; c++)
			{
				// Get a copy of the shared weight vector variable for the submodel
				wModel[c] = w[c].GetCopyFor(model).Named("wModel_" + c + s);
			}
			// Loop over data
			VariableArray<int> ytest = Variable.Array<int>(item).Named("ytest" + s);
			using (Variable.ForEach(item))
			{
				// The score for this item across all components
				Variable<double>[] score = BPMUtils.ComputeClassScores(wModel, xValues[item], NoisePrec);
				// The constraints on the output variable
				ytest[item] = Variable.DiscreteUniform(nClass);
				BPMUtils.ConstrainMaximum(ytest[item], score, nClass);
			}
			// Store the variables 
			BPMVarsModelForTest bpmVar = new BPMVarsModelForTest();
			bpmVar.ie = new InferenceEngine();
			bpmVar.xValues = xValues;
			bpmVar.y = ytest;
			bpmVar.nItems = nItem;
			bpmVar.model = model;
			return bpmVar;
		}

		/// <summary>
		/// Trains the specified submodel for this shraed variable Bayes point machine 
		/// </summary>
		/// <param name="xValuesData">Lists of vectors for each component</param>
		/// <param name="chunkNo">The chunk number</param>
		/// <returns>The posterior distributions for the weight vectors for each component</returns>
		public VectorGaussian[] Train(List<Vector>[] xValuesData, int chunkNo)
		{
			// Set the observed data
			for (int c = 0; c < nClass; c++)
			{
				trainModel.nItems[c].ObservedValue = xValuesData[c].Count;
				trainModel.xValues[c].ObservedValue = xValuesData[c].ToArray();
			}
			// Perform inference for this chunk
			trainModel.model.InferShared(trainModel.ie, chunkNo);
			VectorGaussian[] wInferred = new VectorGaussian[nClass];
			for (int c = 0; c < nClass; c++)
			{
				wInferred[c] = w[c].Marginal<VectorGaussian>();
			}
			// Return the inferred weight vector
			return wInferred;
		}

		/// <summary>
		/// Tests the shared variable Bayes point machine given a chunk number
		/// </summary>
		/// <param name="xValuesData">Lists of vectors for each component</param>
		/// <param name="chunkNo">The chunk number</param>
		/// <returns>The posterior distributions over the classes for each data point</returns>
		public Discrete[] Test(Vector[] xValuesData, int chunkNo)
		{
			// Set the observed data
			testModel.nItems.ObservedValue = xValuesData.Length;
			testModel.xValues.ObservedValue = xValuesData;
			for (int c = 0; c < nClass; c++)
			{
				w[c].SetInput(testModel.model, chunkNo);
			}
			// Infer the outputs
			Discrete[] yInferred = Distribution.ToArray<Discrete[]>(testModel.ie.Infer(testModel.y));
			return yInferred;
		}
	}

}
