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
	public class BPMModelVarsForTrain
	{
		public Model model;
		public BPMDataVars[] dataVars;  // dataVars[k]: variables for kth component
		public InferenceEngine ie;
	}

	/// <summary>
	/// Sparse Bayes point machine - model variables for testing
	/// </summary>
	public class BPMModelVarsForTest
	{
		public Model model;
		public BPMDataVars dataVars;
		public VariableArray<int> y;
		public InferenceEngine ie;

	}

	/// <summary>
	/// Sparse Bayes point machine using shared variables
	/// </summary>
	public class BPMSparse_Shared
	{
		BPMModelVarsForTrain trainModel;
		BPMModelVarsForTest testModel;
		SharedVariableArray<double>[] w;
		int nComponents, nFeatures;
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
		/// Constructs a multi-component sparse Bayes Point Machine using shared variables for chunking data
		/// </summary>
		/// <param name="nClass">Number of components (classes)</param>
		/// <param name="featureCount">Number of features</param>
		/// <param name="noisePrec">Noise precision</param>
		/// <param name="trainChunkSize">Chunk size for training</param>
		/// <param name="testChunkSize">Chunk size for testing</param>
		public BPMSparse_Shared(int nClass, int featureCount, double noisePrec, int trainChunkSize, int testChunkSize)
		{
			nComponents = nClass;
			nFeatures = featureCount;
			NoisePrec = noisePrec;
			this.trainChunkSize = trainChunkSize;
			this.testChunkSize = testChunkSize;
			feature = new Range(nFeatures).Named("feature");
			w = new SharedVariableArray<double>[nComponents];
			IDistribution<double[]> wPrior0 = Distribution<double>.Array(nFeatures,
				delegate(int index) { return Gaussian.PointMass(0); });
			IDistribution<double[]> wPrior = Distribution<double>.Array(nFeatures,
				delegate(int index) { return Gaussian.FromMeanAndPrecision(0.0, 1.0); });
			for (int c = 0; c < nComponents; c++)
			{
				w[c] = (c == 0)
					? SharedVariable<double>.Random(feature, (DistributionStructArray<Gaussian,double>)wPrior0).Named("w_" + c)
					: SharedVariable<double>.Random(feature, (DistributionStructArray<Gaussian,double>)wPrior).Named("w_" + c);
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
		private BPMModelVarsForTrain SpecifyTrainModel(string s, int nChunks)
		{
			BPMDataVars[] dataVars = new BPMDataVars[nComponents];
			// The model identifier for the shared variables
			Model model = new Model(nChunks).Named("model" + s);
			// The weight vector within a submodel
			VariableArray<double>[] wModel = new VariableArray<double>[nComponents];
			for (int c = 0; c < nComponents; c++)
			{
				// Get a copy of the shared weight vector variable for the submodel
				wModel[c] = w[c].GetCopyFor(model).Named("wModel_" + c + s);
			}
			for (int c = 0; c < nComponents; c++)
			{
				Variable<int> nItem = Variable.New<int>().Named("nItem_" + c + s);
				Range item = new Range(nItem).Named("item_" + c + s);
				VariableArray<int> xValueCount = Variable.Array<int>(item).Named("xCount_" + c + s);
				Range itemFeature = new Range(xValueCount[item]).Named("itemFeature_" + c + s);
				VariableArray<VariableArray<double>, double[][]> xValues = Variable.Array(Variable.Array<double>(itemFeature), item).Named("xValues_" + c + s);
				VariableArray<VariableArray<int>, int[][]> xIndices = Variable.Array(Variable.Array<int>(itemFeature), item).Named("xIndices_" + c + s);
				using (Variable.ForEach(item))
				{
					// The score for this item across all components
					Variable<double>[] score =BPMUtils.ComputeClassScores(wModel, xValues[item], xIndices[item], itemFeature, NoisePrec);
					BPMUtils.ConstrainArgMax(c, score);
				}
				dataVars[c] = new BPMDataVars(nItem, item, xIndices, xValueCount, xValues);

			}

			BPMModelVarsForTrain bpmVar = new BPMModelVarsForTrain();
			bpmVar.ie = new InferenceEngine();
			bpmVar.dataVars = new BPMDataVars[nComponents];
			bpmVar.model = model;
			bpmVar.dataVars = dataVars;
			return bpmVar;
		}

		/// <summary>
		/// Specify the training model
		/// </summary>
		/// <param name="s">The name of the test model</param>
		/// <param name="nChunks">The number of chunks</param>
		/// <returns>A <see cref="BPMVarsModelForTest"/> instance</returns>
		private BPMModelVarsForTest SpecifyTestModel(string s, int nChunks)
		{
			Variable<int> nItem = Variable.New<int>().Named("nItem_" + s);
			Range item = new Range(nItem).Named("item_" + s);
			VariableArray<int> xValueCount = Variable.Array<int>(item).Named("xCount_" + s);
			Range itemFeature = new Range(xValueCount[item]).Named("itemFeature" + s);
			VariableArray<VariableArray<double>, double[][]> xValues = Variable.Array(Variable.Array<double>(itemFeature), item).Named("xValues" + s);
			VariableArray<VariableArray<int>, int[][]> xIndices = Variable.Array(Variable.Array<int>(itemFeature), item).Named("xIndices" + s);

			Model model = new Model(nChunks).Named("model" + s);
			VariableArray<double>[] wModel = new VariableArray<double>[nComponents];
			for (int c = 0; c < nComponents; c++)
			{
				wModel[c] = w[c].GetCopyFor(model).Named("model_" + c + s);
			}

			VariableArray<int> ytest = Variable.Array<int>(item).Named("ytest" + s);
			using (Variable.ForEach(item))
			{
				// The score for this item across all components
				Variable<double>[] score = BPMUtils.ComputeClassScores(wModel, xValues[item], xIndices[item], itemFeature, NoisePrec);
				ytest[item] = Variable.DiscreteUniform(nComponents);
				BPMUtils.ConstrainMaximum(ytest[item], score, nComponents);
			}
			BPMModelVarsForTest bpmVar = new BPMModelVarsForTest();
			bpmVar.ie = new InferenceEngine();
			bpmVar.dataVars = new BPMDataVars(nItem, item, xIndices, xValueCount, xValues);
			bpmVar.y = ytest;
			bpmVar.model = model;
			return bpmVar;
		}

		/// <summary>
		/// Test the specified submodel for this shared variable sparse Bayes point machine 
		/// </summary>
		/// <param name="chunkIndex">Index of the chunk</param>
		/// <param name="xIndicesData">Jagged array of data indices in chunk, chunkIndex. xIndicesData[t]: feature indices for t^{th} datapoint </param>
		/// <param name="xValuesData">Jagged array of data values   in chunk, chunkIndex.  xValuesData[t]: feature values for t^{th} datapoint</param>
		/// <returns></returns>
		public Discrete[] Test(int[][] xIndicesData, double[][] xValuesData, int chunkIndex)
		{
			testModel.dataVars.SetObservedValues(xIndicesData, xValuesData);
			for (int c = 0; c < nComponents; c++)
			{
				w[c].SetInput(testModel.model, chunkIndex);
			}
			return Distribution.ToArray<Discrete[]>(testModel.ie.Infer(testModel.y));
		}

		/// <summary>
		/// Trains the specified submodel for this shared variable sparse Bayes point machine 
		/// </summary>
		/// <param name="chunkIndex">Index of the chunk</param>
		///  <param name="xIndicesData">Jagged array of indices for all classes  in chunk, chunkIndex; xIndicesData[k][t]: feature indices for t^{th} datapoint of k^{th} class</param>
		/// <param name="xValuesData">Jagged array of data for all classes  in chunk, chunkIndex; xValuesData[k][t]: feature values for t^{th} datapoint of k^{th} class</param>
		/// <returns>The posterior distributions for the weight vectors for each component</returns>
		public Gaussian[][] Train(int[][][] xIndicesData, double[][][] xValuesData, int chunkIndex)
		{
			for (int c = 0; c < nComponents; c++)
			{
				trainModel.dataVars[c].SetObservedValues(xIndicesData[c], xValuesData[c]);
			}
			trainModel.model.InferShared(trainModel.ie, chunkIndex);

			Gaussian[][] wInferred = new Gaussian[nComponents][];
			for (int c = 0; c < nComponents; c++)
			{
				wInferred[c] = Distribution.ToArray<Gaussian[]>(w[c].Marginal<IDistribution<double[]>>());
			}
			return wInferred;
		}
	}

}
