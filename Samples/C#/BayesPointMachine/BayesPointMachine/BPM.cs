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
	/// Bayes point machine - model variables for training
	/// </summary>
	public class BPMVarsForTrain
	{
		public InferenceEngine ie;
		/// <summary>
		/// The number of items in each class
		/// </summary>
		public Variable<int>[] nItems;
		/// <summary>
		/// The feature vectors for each class; xValues[c] is a variable array of size nItems[c]
		/// </summary>
		public VariableArray<Vector>[] xValues;
		/// <summary>
		/// The weight vectors for each class
		/// </summary>
		public Variable<Vector>[] w;
		/// <summary>
		/// The initial priors for the weight vectors for each class
		/// </summary>
		public Variable<VectorGaussian>[] wInit;
	}

	/// <summary>
	/// Bayes point machine - model variables for testing
	/// </summary>
	class BPMVarsForTest
	{
		public InferenceEngine ie;
		/// <summary>
		/// Number of items
		/// </summary>
		public Variable<int> nItems;
		/// <summary>
		/// Feature vectors
		/// </summary>
		public VariableArray<Vector> xValues;
		/// <summary>
		/// The output of the model
		/// </summary>
		public VariableArray<int> y;
		/// <summary>
		/// The weight vectors for each class
		/// </summary>
		public Variable<Vector>[] w;
		/// <summary>
		/// The priors for the weight vectors for each class
		/// </summary>
		public Variable<VectorGaussian>[] wPrior;
	}

	/// <summary>
	/// Multi-component Bayes point machine
	/// </summary>
	public class BPM
	{
		BPMVarsForTrain trainModel;
		BPMVarsForTest testModel;
		int nClass, nFeatures;

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
		/// Constructs an instance from number of components and number of features
		/// </summary>
		/// <param name="nClass">Number of classes</param>
		/// <param name="nFeatures">Number of features</param>
		/// <param name="noisePrec">Noise precision</param>
		public BPM(int nClass, int nFeatures, double noisePrec)
		{
			this.nClass = nClass;
			this.nFeatures = nFeatures;
			NoisePrec = noisePrec;
			trainModel = SpecifyTrainModel("_train");
			testModel = SpecifyTestModel("_test");
		}

		/// <summary>
		/// Specifies the training model
		/// </summary>
		/// <param name="s">The name of the training model</param>
		/// <returns>A <see cref="BPMVarsForTrain"/> instance</returns>
		private BPMVarsForTrain SpecifyTrainModel(string s)
		{
			// An array of feature vectors - their observed values will be
			// set by the calling program
			VariableArray<Vector>[] xValues = new VariableArray<Vector>[nClass];
			// The number of items for each component
			Variable<int>[] nItem = new Variable<int>[nClass];
			// Ranges over the items for each component
			Range[] item = new Range[nClass];
			// The weight vector for each component
			Variable<Vector>[] w = new Variable<Vector>[nClass];
			// The prior weight distributions for each component
			Variable<VectorGaussian>[] wInit = new Variable<VectorGaussian>[nClass];
			for (int c = 0; c < nClass; c++)
			{
				// The prior distributions will be set by the calling program
				wInit[c] = Variable.New<VectorGaussian>();
				// The weight vectors are drawn from the prior distribution
				w[c] = Variable<Vector>.Random(wInit[c]);
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
				// Loop over the items
				using (Variable.ForEach(item[c]))
				{
					// The score for this item across all components
					Variable<double>[] score = BPMUtils.ComputeClassScores(w, xValues[c][item[c]], NoisePrec);
					// The constraint imposed by the observed component
					BPMUtils.ConstrainArgMax(c, score);
				}
			}
			// Store the variables 
			BPMVarsForTrain bpmVar = new BPMVarsForTrain();
			bpmVar.ie = new InferenceEngine();
			bpmVar.xValues = xValues;
			bpmVar.nItems = nItem;
			bpmVar.w = w;
			bpmVar.wInit = wInit;
			return bpmVar;
		}

		/// <summary>
		/// Specifies the test model
		/// </summary>
		/// <param name="s">The name of the test model</param>
		/// <returns>A <see cref="BPMVarsForTest"/> instance</returns>
		private BPMVarsForTest SpecifyTestModel(string s)
		{
			// The number of test items - this will be set by the calling program
			Variable<int> nItem = Variable.New<int>().Named("nItem" + s);
			// A range over the items
			Range item = new Range(nItem).Named("item" + s);
			// An array of feature vectors - their observed values will be
			// set by the calling program
			VariableArray<Vector> xValues = Variable.Array<Vector>(item);
			// The weight vectors for each component
			Variable<Vector>[] w = new Variable<Vector>[nClass];
			// The prior distribution for weight vector for each component. When
			// <see cref="Test"/> is called, this is set to the posterior weight
			// distributions from <see cref="Train"/>
			Variable<VectorGaussian>[] wPrior = new Variable<VectorGaussian>[nClass];
			// Loop over the classes
			for (int c = 0; c < nClass; c++)
			{
				// The priors will be set by the calling program
				wPrior[c] = Variable.New<VectorGaussian>();
				// The weights are sampled from the prior distributions
				w[c] = Variable<Vector>.Random(wPrior[c]);
			}
			// Loop over the data 
			VariableArray<int> ytest = Variable.Array<int>(item).Named("ytest" + s);
			using (Variable.ForEach(item))
			{
				// The score for this item across all components
				Variable<double>[] score = BPMUtils.ComputeClassScores(w, xValues[item], NoisePrec);
				// The constraints on the output variable
				ytest[item] = Variable.DiscreteUniform(nClass);
				BPMUtils.ConstrainMaximum(ytest[item], score, nClass);
			}
			// Store the variables 
			BPMVarsForTest bpmVar = new BPMVarsForTest();
			bpmVar.ie = new InferenceEngine();
			bpmVar.xValues = xValues;
			bpmVar.y = ytest;
			bpmVar.nItems = nItem;
			bpmVar.w = w;
			bpmVar.wPrior = wPrior;
			return bpmVar;
		}

		/// <summary>
		/// Incrementally trains this Bayes point machine
		/// </summary>
		/// <param name="xValuesData">Lists of vectors for each component</param>
		/// <returns>The posterior distributions for the weight vectors for each component</returns>
		public VectorGaussian[] TrainIncremental(List<Vector>[] xValuesData)
		{
			return ((trainModel.wInit[0].IsObserved) ? InferW(xValuesData) : Train(xValuesData));
		}

		/// <summary>
		/// Trains this Bayes point machine
		/// </summary>
		/// <param name="xValuesData">Lists of vectors for each component</param>
		/// <returns>The posterior distributions for the weight vectors for each component</returns>
		public VectorGaussian[] Train(List<Vector>[] xValuesData)
		{
			for (int c = 0; c < nClass; c++)
			{
				trainModel.wInit[c].ObservedValue = (c == 0)
					? VectorGaussian.PointMass(Vector.Zero(nFeatures))
					: VectorGaussian.FromMeanAndPrecision(Vector.Zero(nFeatures), PositiveDefiniteMatrix.Identity(nFeatures));
			}
			return InferW(xValuesData);
		}

		/// <summary>
		/// Performs inference on this Bayes point machine
		/// </summary>
		/// <param name="xValuesData">Lists of vectors for each component</param>
		/// <returns></returns>
		private VectorGaussian[] InferW(List<Vector>[] xValuesData)
		{
			// Set the observed data
			for (int c = 0; c < nClass; c++)
			{
				trainModel.nItems[c].ObservedValue = xValuesData[c].Count;
				trainModel.xValues[c].ObservedValue = xValuesData[c].ToArray();
			}
			// Infer the weights
			VectorGaussian[] wInfer = new VectorGaussian[nClass];
			// Reset the priors to support incremental training
			for (int c = 0; c < nClass; c++)
			{
				wInfer[c] = (trainModel.ie).Infer<VectorGaussian>(trainModel.w[c]);
				trainModel.wInit[c].ObservedValue = wInfer[c];
			}
			return wInfer;
		}

		/// <summary>
		/// Tests this Bayes point machine
		/// </summary>
		/// <param name="xValuesData">Lists of vectors for each component</param>
		/// <returns>The posterior distributions over the components for each data point</returns>
		public Discrete[] Test(Vector[] xValuesData)
		{
			// Set the observed data
			for (int c = 0; c < nClass; c++)
			{
				testModel.wPrior[c].ObservedValue = trainModel.wInit[c].ObservedValue;
			}
			testModel.nItems.ObservedValue = xValuesData.Length;
			testModel.xValues.ObservedValue = xValuesData;

			// Infer the outputs
			Discrete[] yInferred = Distribution.ToArray<Discrete[]>(testModel.ie.Infer(testModel.y));
			return yInferred;
		}
	}

}
