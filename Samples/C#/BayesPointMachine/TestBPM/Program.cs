//#define ShowWeights
using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Transforms;
using System.IO;

namespace BayesPointMachine
{
	class Program
	{
		static void Main(string[] args)
		{
			int nClass = 3;
			int nItems = 30;
			int totalFeatures = 4;
			int maxItemsInBatch = 10;
			double noisePrec = 0.1;
			int nChunks = ((nItems - 1) / maxItemsInBatch )+1;
			string trainingFile = @"..\..\data\data.txt";

			Vector[] testData = new Vector[2];
			testData[0] = Vector.FromArray(new double[] { 2.1, 0, 0, 0 });
			testData[1] = Vector.FromArray(new double[] { 0, 0, 1.3, 0 });
			int[][] indicesTestData = new int[2][];
			double[][] valuesTestData = new double[2][];
			indicesTestData[0] = new int[] { 0 };
			indicesTestData[1] = new int[] { 2 };
			valuesTestData[0] = new double[] { 2.1 };
			valuesTestData[1] = new double[] { 1.3 };

			Test_BPM(nClass, totalFeatures, noisePrec, trainingFile, testData);
			Test_BPMIncremental(nClass, totalFeatures, noisePrec, maxItemsInBatch, nChunks, trainingFile, testData);
			Test_BPM_Shared(nClass, totalFeatures, noisePrec, maxItemsInBatch, nChunks, trainingFile, testData);
			Test_BPM_Sparse(nClass, totalFeatures, noisePrec, trainingFile, indicesTestData, valuesTestData);
			Test_BPM_Sparse_Shared(nClass, totalFeatures, noisePrec, maxItemsInBatch, nChunks, trainingFile, indicesTestData, valuesTestData);

			Console.WriteLine("Press Enter to quit");
			Console.ReadKey();
		}

		private static void Test_BPM(int nClass, int totalFeatures, double noisePrec, string fileName, Vector[] testData)
		{
			Console.WriteLine("\n------- BPM -------");
			List<Vector>[] data = DataFromFile.Read(fileName, nClass);
			BPM bpm = new BPM(nClass, totalFeatures, noisePrec);
			bpm.TrainingEngine.ShowProgress = false;
			bpm.TestEngine.ShowProgress = false;
			VectorGaussian[] wInfer = bpm.Train(data);
#if ShowWeights
			for (int i = 0; i < wInfer.GetLength(0); i++)
			{
				Console.WriteLine(wInfer[i].ToString());
			}
#endif
			Discrete[] predictions = bpm.Test(testData);
			Console.WriteLine("\nPredictions:");
			foreach (Discrete pred in predictions)
				Console.WriteLine(pred);
			Console.WriteLine();
		}

		private static void Test_BPMIncremental(
			int nClass, int totalFeatures, double noisePrec, int maxItemsInBatch,
			int nChunks, string trainingFile, Vector[] testData)
		{
			Console.WriteLine("\n------- BPM Train Incremental -------");
			VectorGaussian[] wInfer = new VectorGaussian[nClass];
			BPM bpmIncremental = new BPM(nClass, totalFeatures, noisePrec);
			bpmIncremental.TrainingEngine.ShowProgress = false;
			bpmIncremental.TestEngine.ShowProgress = false;
			int LocToStart = 0;
			for (int c = 0; c < nChunks; c++)
			{
				List<Vector>[] dataChunks = DataFromFile.Read(trainingFile, nClass, maxItemsInBatch, ref LocToStart);
				wInfer = bpmIncremental.TrainIncremental(dataChunks);
			}
#if ShowWeights
			for (int i = 0; i < wInfer.GetLength(0); i++)
			{
				Console.WriteLine(wInfer[i].ToString());
			}
#endif
			Console.WriteLine("\nPredictions:");
			Discrete[] predictions = bpmIncremental.Test(testData);
			foreach (Discrete pred in predictions)
				Console.WriteLine(pred);
			Console.WriteLine();
		}

		private static void Test_BPM_Shared(int nClass, int totalFeatures, double noisePrec, int maxItemsInBatch, int nChunks, string trainingFile, Vector[] testData)
		{
			Console.WriteLine("\n------- BPM Shared -------");

			VectorGaussian[] wInfer = new VectorGaussian[nClass];
			BPM_Shared bpmShared = new BPM_Shared(nClass, totalFeatures, noisePrec, nChunks, 1);
			bpmShared.TrainingEngine.ShowProgress = false;
			bpmShared.TestEngine.ShowProgress = false;
			// Several passes to achieve convergence
			for (int pass = 0; pass < 15; pass++)
			{
				int LocToStart = 0;
				// Loop over chunks
				for (int c = 0; c < nChunks; c++)
				{
					List<Vector>[] dataChunks = DataFromFile.Read(trainingFile, nClass, maxItemsInBatch, ref LocToStart);
					wInfer = bpmShared.Train(dataChunks, c);
				}
			}
#if ShowWeights
			for (int i = 0; i < wInfer.GetLength(0); i++)
			{
				Console.WriteLine(wInfer[i]);
			}
#endif
			Console.WriteLine("\nPredictions:");
			Discrete[] predictions = bpmShared.Test(testData, 0);
			foreach (Discrete pred in predictions)
				Console.WriteLine(pred);
			Console.WriteLine();
		}

		private static void Test_BPM_Sparse(
			int nClass, int totalFeatures, double noisePrec, string trainingFile,
			int[][] xIndicesTest, double[][] xValuesTest)
		{
			Console.WriteLine("\n------- BPM Sparse -------");
			Gaussian[][] wInferred;
			int[][][] xIndices;
			double[][][] xValues = DataFromFile.Read(trainingFile, nClass, out xIndices);
			BPMSparse bpmSparse = new BPMSparse(nClass, totalFeatures, noisePrec);
			bpmSparse.TrainingEngine.ShowProgress = false;
			bpmSparse.TestEngine.ShowProgress = false;
			//if want to see the Browser, uncomment this line
			// bpmSparse.TrainingEngine.BrowserMode = BrowserMode.Always;
			wInferred = bpmSparse.Train(xIndices, xValues);
#if ShowWeights
			for (int i = 0; i < wInferred.GetLength(0); i++)
			{
				for (int j = 0; j < wInferred[i].Length; j++)
				{
					Console.WriteLine(wInferred[i][j].ToString());
				}

			}
#endif
			Console.WriteLine("\nPredictions:");
			Discrete[] predictions = bpmSparse.Test(xIndicesTest, xValuesTest);
			foreach (Discrete pred in predictions)
				Console.WriteLine(pred);
			Console.WriteLine();
		}

		private static void Test_BPM_Sparse_Shared(
			int nClass, int totalFeatures, double noisePrec, int maxItemsInBatch, int nChunks, string trainingFile,
			int[][] xIndicesTest, double[][] xValuesTest)
		{
			Console.WriteLine("\n------- BPM Sparse Shared -------");
			BPMSparse_Shared bpmSparseShared = new BPMSparse_Shared(nClass, totalFeatures, noisePrec, nChunks, 1);
			bpmSparseShared.TrainingEngine.ShowProgress = false;
			bpmSparseShared.TestEngine.ShowProgress = false;
			Gaussian[][] wInferred = new Gaussian[nClass][];
			for (int c = 0; c < nClass; c++)
			{
				wInferred[c] = new Gaussian[totalFeatures];
			}
			for (int pass = 0; pass < 5; pass++)
			{
				int LocToStart = 0;
				for (int c = 0; c < nChunks; c++)
				{
					int[][][] xIndices;
					double [][][] xValues = DataFromFile.Read(trainingFile, nClass, maxItemsInBatch, ref LocToStart, out xIndices);
					wInferred = bpmSparseShared.Train(xIndices, xValues, c);
				}
			}
#if ShowWeights
			for (int i = 0; i < wInferred.GetLength(0); i++)
			{
				for (int j = 0; j < wInferred[i].Length; j++)
				{
					Console.WriteLine(wInferred[i][j].ToString());
				}
			}
#endif
			Console.WriteLine("\nPredictions:");
			Discrete[] predictions = bpmSparseShared.Test(xIndicesTest, xValuesTest, 0);
			foreach (Discrete pred in predictions)
				Console.WriteLine(pred);
			Console.WriteLine();
		}
	}

}