using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace ImageClassifier
{
	static class Program
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main()
		{
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
#if VS_V8
			Application.Run(new Form1());
#else
			ItemsModel model = new ItemsModel();
			model.PopulateFromStringsAndVectors(Form1.ReadLines(model.form1.folder + "Images.txt"), model.form1.data);
			ClassifierView cv = new ClassifierView();
			cv.DataContext = model;
			cv.ShowInForm("Image Classifer using Infer.NET");
#endif
		}
	}
}