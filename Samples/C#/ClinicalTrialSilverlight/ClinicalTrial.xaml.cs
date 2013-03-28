using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using MicrosoftResearch.Infer.Distributions;
using System.Collections.ObjectModel;
using System.Windows.Controls.Primitives;
using System.IO;
using System.Windows.Media.Animation;

namespace ClinicalTrial
{
	/// <summary>
	/// Interaction logic for ClinicalTrial.xaml
	/// </summary>
	public partial class MainWindow : UserControl
	{
		public ClinicalTrialGeneratedCode model = null;
		public int[] counts = new int[] { 0, 0, 0, 0 };

		public MainWindow()
		{
			InitializeComponent();
			model = new ClinicalTrialGeneratedCode();
			infer(new bool[0], new bool[0]);
		}

		private static void drawBetaDistribution(ListBox lb, Beta dist)
		{
			lb.Items.Clear();
			int numItems = (int)(lb.ActualWidth/3.0);
			double max = 6.0;
			double mult = ((double)lb.ActualHeight) / max;
			double inc = 1.0 / ((double)(numItems-1));

			double curr = 0.0;
			for (int i=0; i < numItems; i++)
			{
				if (curr > 1.0)
					curr = 1.0;
				double d = Math.Exp(dist.GetLogProb(curr));
				double height = mult * d;
				lb.Items.Add(new Rectangle() { Height=height, Width=2, Fill= Yellow, VerticalAlignment= VerticalAlignment.Bottom});
				curr += inc;
			}
		}
		public static Brush Yellow = new SolidColorBrush(Colors.Yellow);

		private static void drawBetaDistribution(Rectangle rect, Beta dist)
		{
			GradientStopCollection gsc = new GradientStopCollection();
			int numStops = 21;
			double mean = dist.GetMean();
			double meanDensity = Math.Exp(dist.GetLogProb(mean));
			double inc = 1.0 / (numStops-1);
			double curr = 0.0;
			double maxLogProb = Double.MinValue;
			double minLogProb = -5.0;
			for (int i=0; i < numStops; i++)
			{
				double logProb = dist.GetLogProb(curr);
				if (logProb > maxLogProb) maxLogProb = logProb;
				curr += inc;
			}
			if (maxLogProb <= minLogProb)
				maxLogProb = minLogProb + 1.0;
			double diff = maxLogProb - minLogProb;
			double mult =  1.0 / (maxLogProb - minLogProb);
			curr = 0.0;
			double blueLeft = 0; double blueRight = 0;
			double redLeft = 255; double redRight = 255;
			double greenLeft = 255; double greenRight = 255;

			for (int i=0; i < numStops; i++)
			{
				double red, green, blue;
				double logProb = dist.GetLogProb(curr);
				if (logProb < minLogProb) logProb = minLogProb;
				double level = mult * (logProb - minLogProb);
				red = level * (mean * redRight + (1.0 - mean) * redLeft);
				green = level * (mean * greenRight + (1.0 - mean) * greenLeft);
				blue =level * (mean * blueRight + (1.0 - mean) * blueLeft);
				byte redb = red < 0.0 ? (byte)0 : red > 255.0 ? (byte)255 : (byte)red;
				byte greenb = green < 0.0 ? (byte)0 : green > 255.0 ? (byte)255 : (byte)green;
				byte blueb = blue < 0.0 ? (byte)0 : blue > 255.0 ? (byte)255 : (byte)blue;
				gsc.Add(new GradientStop { Color =new Color() { A = 255, R = redb, G = greenb, B = blueb } , Offset = curr});
    			curr += inc;
			}
			LinearGradientBrush brush = rect.Fill as LinearGradientBrush;
			brush.GradientStops = gsc;
		}

		private void infer(bool[] treated, bool[] placebo)
		{
			model.numberPlacebo = placebo.Length;
			model.numberTreated = treated.Length;
			model.treatedGroupOutcomes = treated;
			model.placeboGroupOutcomes = placebo;
			model.Execute(50);


			double probTrue = model.IsEffectiveMarginal().GetProbTrue(); 
			//ProbIsEffectiveSlider.Value = 
			ProbRect.Height =Math.Max(0, (ProbIsEffectiveSlider.ActualHeight - 20) * probTrue);
			drawBetaDistribution(TreatedPDF, model.ProbIfTreatedMarginal());
			drawBetaDistribution(PlaceboPDF, model.ProbIfPlaceboMarginal());
		}

		private void UpdateList(double val, ListBox lb, int index)
		{
			// Following assumes 20 items max.
			int new_count = (int)val;
			int cur_count = counts[index];
			// If count has changed...
			if (new_count != cur_count)
			{
				// ... update the list box ...
				lb.Items.Clear();
				for (int i=0; i < new_count; i++)
				{
					// the actual object added here does not matter (it will be ignored)
					// the size of the collection is all that matters
					lb.Items.Add("patient"); 
				}
				counts[index] = new_count;
				// ... construct the true/false arrays...
				int numTreated = counts[0] + counts[2];
				int numPlacebo = counts[1] + counts[3];
				bool[] treated = new bool[numTreated];
				bool[] placebo = new bool[numPlacebo];
				int j=0;
				for (; j < counts[0]; j++)
					treated[j] = true;
				for (; j < numTreated; j++)
					treated[j] = false;
				j=0;
				for (; j < counts[1]; j++)
					placebo[j] = true;
				for (; j < numPlacebo; j++)
					placebo[j] = false;
				// ... and do the inference
				infer(treated, placebo);
			}

		}

		FrameworkElement startEl;
		bool down = false;
		private void Rectangle_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
		{
			down = true;
			FrameworkElement el = (FrameworkElement)sender;
			startEl = el;
			Point pos = e.GetPosition(el);
			// Size of a child element
			pos = OnClickOrDrag(el, pos);
			e.Handled = true;
		}

		Size childSize = new Size(22+2*3, 34+8);
		private Point OnClickOrDrag(FrameworkElement el, Point pos)
		{
			// Size of a child element
			int numPerRow = 6;// (int)(el.ActualWidth / childSize.Width);
			int col =(int)((pos.X + 20+4) / childSize.Width); col = Math.Min(col, 6);
			int row =(int)(pos.Y / childSize.Height); row = Math.Min(row, 6);
			if ((string)el.Tag == "TreatedGood") UpdateList(row*numPerRow + col, ListBoxTreatedGood, 0);
			if ((string)el.Tag == "PlaceboGood") UpdateList(row*numPerRow + col, ListBoxPlaceboGood, 1);
			if ((string)el.Tag == "TreatedBad") UpdateList(row*numPerRow + col, ListBoxTreatedBad, 2);
			if ((string)el.Tag == "PlaceboBad") UpdateList(row*numPerRow + col, ListBoxPlaceboBad, 3);
			return pos;
		}

		private void Rectangle_MouseMove(object sender, MouseEventArgs e)
		{
			if (!down) return;
			//if (e.LeftButton != MouseButtonState.Pressed) return;

			FrameworkElement el = (FrameworkElement)sender;
			if (el!=startEl) return;
			Point pos = e.GetPosition(el);
			pos = OnClickOrDrag(el, pos);
			//e.Handled = true;
		}

		private void Reset_Clicked(object sender, RoutedEventArgs e)
		{
			UpdateList(0, ListBoxTreatedGood, 0);
			UpdateList(0, ListBoxPlaceboGood, 1);
			UpdateList(0, ListBoxTreatedBad, 2);
			UpdateList(0, ListBoxPlaceboBad, 3);
		}


		static Random rnd = new Random();
		private FrameworkElement GetRandomElement(ListBox lb)
		{
			FrameworkElement el=null;
			if (lb.Items.Count>0)
			{
				for (int i=0; i<20; i++)
				{
					int ind = rnd.Next(lb.Items.Count);
					el = (FrameworkElement)lb.ItemContainerGenerator.ContainerFromIndex(ind);
					if (el.Tag!=null) el=null;
					if (el!=null) break;
				}
			}
			return el;
		}

		private void Rectangle_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
		{
			down = false;
		}

		private void RotatePerson(ListBox lb)
		{
			FrameworkElement el = GetRandomElement(lb);
			if (el!=null)
			{
				el.Tag="Dead";
				RotateTransform rt = new RotateTransform { CenterX = childSize.Width/2, CenterY = childSize.Height -14 };
				el.RenderTransform = rt;
				Storyboard sb = new Storyboard();
				var da = new DoubleAnimation { To=90, Duration= new Duration(TimeSpan.FromMilliseconds(300)) };
				da.SetValue(Storyboard.TargetPropertyProperty, new PropertyPath("Angle"));
				Storyboard.SetTarget(da, rt);
				sb.Children.Add(da);
				sb.Begin();
			}
		}

		private void UserControl_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
		{
			FrameworkElement el = sender as FrameworkElement;
			if (el==null) return;
			RotatePerson((el.Tag as string) == "PlaceboBad" ? ListBoxPlaceboBad : ListBoxTreatedBad);
			e.Handled = true;
		}

		private void Rectangle_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
		{
			e.Handled = true;
		}

	
	}
}
