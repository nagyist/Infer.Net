// (C) Copyright 2008 Microsoft Research Cambridge
using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Collections;
using System.IO;

namespace BayesPointMachine
{


	/// <summary>
	/// Class to read the file in the format <classID, feature 1 value, feature 2 value, .... feature N value>
	/// Assumes that if the value for a feature is valueToIgnore, then this value is to be ignored
	/// </summary>
	/// 
	class DataFromFile
	{
		public static double valueToIgnore = 0;



		/// <summary>
		/// Reads data from file 
		/// </summary>
		/// <param name="filename">The name of the file to read from</param>
		/// <param name="nClass">The name of possible classes in the file</param>
		/// <param name="maxItemsInBatch">Maximum number of data points to read from the file</param>
		/// <param name="LocationToStart">Location to start reading the file</param>
		/// <returns> jagged array of data values and data indices read from the file</returns>
		/// 

		public static double[][][] Read(
			string filename, int nClass, int maxItemsInBatch,
			ref int LocationToStart, out int[][][] indices)
		{
			List<Vector>[] data = Read(filename, nClass, maxItemsInBatch, ref  LocationToStart);
			return MakeDataSparse(data, nClass, out indices);
		}


		/// <summary>
		/// Reads data from file 
		/// </summary>
		/// <param name="filename">The name of the file to read from</param>
		/// <param name="nClass">The name of possible classes in the file</param>
		/// <returns> jagged array of data values and data indices read from the file</returns>
		/// 

		public static double[][][] Read(string filename, int nClass, out int[][][] indices)
		{
			List<Vector>[] data = Read(filename, nClass);

			return MakeDataSparse(data, nClass, out indices);
		}

		private static double[][][] MakeDataSparse(List<Vector>[] data, int nClass, out int[][][] indices)
		{
			double[][][] x = new double[nClass][][];
			indices = new int[nClass][][];
			for (int c = 0; c < nClass; c++)
			{
				int itemCnt = data[c].Count;

				x[c] = new double[itemCnt][];
				indices[c] = new int[itemCnt][];
				for (int t = 0; t < itemCnt; t++)
				{
					List<double> feature = new List<double>();
					List<int> featureIndex = new List<int>();
					for (int f = 0; f < data[c][t].Count; f++)
					{
						if (data[c][t][f] != valueToIgnore)
						{
							feature.Add(data[c][t][f]);
							featureIndex.Add(f);
						}
					}
					x[c][t] = feature.ToArray();
					indices[c][t] = featureIndex.ToArray();
				}
			}
			return x;
		}

		/// <summary>
		/// Reads data from file 
		/// </summary>
		/// <param name="filename">The name of the file to read from</param>
		/// <param name="nClass">The name of possible classes in the file</param>
		/// <returns> list of vectors of data for each class</returns>
		public static List<Vector>[] Read(string filename, int nClass)
		{
			int LocToStart = 0;
			return Read(filename, nClass, Int32.MaxValue, ref LocToStart);
		}


		/// <summary>
		/// Reads data from file 
		/// </summary>
		/// <param name="filename">The name of the file to read from</param>
		/// <param name="nClass">The name of possible classes in the file</param>
		/// <param name="maxItemsInBatch">Maximum number of data points to read from the file</param>
		/// <param name="LocationToStart">Location to start reading the file</param>
		/// <returns> list of vectors of data for each class</returns>
		/// 
		public static List<Vector>[] Read(string filename, int nClass, int maxItemsInBatch, ref int LocationToStart)
		{
			List<Vector>[] data = new List<Vector>[nClass];
			for (int c = 0; c < nClass; c++)
			{
				data[c] = new List<Vector>();
			}
			int curLoc = 0; string line;
			StreamReader reader = new StreamReader(filename);
			while (!reader.EndOfStream && curLoc < LocationToStart)
			{
				line = reader.ReadLine();
				curLoc = curLoc + 1;
			}
			int dataSoFar = 0;
			while (!reader.EndOfStream && dataSoFar < maxItemsInBatch)
			{
				line = reader.ReadLine();
				string[] pieces = line.Split('\t', ' ', ',');
				double[] x = new double[pieces.Length - 1];
				int classId = Int16.Parse(pieces[0]);
				for (int i = 1; i <= x.Length; i++)
				{
					x[i - 1] = Double.Parse(pieces[i]);
				}
				data[classId].Add(Vector.FromArray(x));
				dataSoFar = dataSoFar + 1;
				curLoc = curLoc + 1;
			}
			LocationToStart = curLoc;
			return data;
		}
	}

}