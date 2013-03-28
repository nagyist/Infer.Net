using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace ClinicalTrial
{
    public class WrapPanel : Panel
    {
        public Orientation Orientation
        {
            get { return (Orientation)GetValue(OrientationProperty); }
            set { SetValue(OrientationProperty, value); }
        }

        public static readonly DependencyProperty OrientationProperty =
            DependencyProperty.Register("Orientation", typeof(Orientation), typeof(WrapPanel), null);

        public WrapPanel()
        {
            // default orientation
            Orientation = Orientation.Horizontal;
        }

        protected override Size MeasureOverride(Size availableSize)
        {
            foreach (UIElement child in Children)
            {
                child.Measure(new Size(availableSize.Width, availableSize.Height));   
            }

            return base.MeasureOverride(availableSize);
        }

        protected override Size ArrangeOverride(Size finalSize)
        {
            Point point = new Point(0,0);

            int i = 0;

            if (Orientation == Orientation.Horizontal)
            {
                double largestHeight = 0.0;

                foreach (UIElement child in Children)
                {
                    child.Arrange(new Rect(point, new Point(point.X + child.DesiredSize.Width, point.Y + child.DesiredSize.Height)));

                    if (child.DesiredSize.Height > largestHeight)
                        largestHeight = child.DesiredSize.Height;

                    point.X = point.X + child.DesiredSize.Width;

                    if ((i + 1) < Children.Count)
                    {
                        if ((point.X + Children[i + 1].DesiredSize.Width) > finalSize.Width)
                        {
                            point.X = 0;
                            point.Y = point.Y + largestHeight;
                            largestHeight = 0.0;
                        }
                    }

                    i++;
                }
            }
            else
            {
                double largestWidth = 0.0;

                foreach (UIElement child in Children)
                {
                    child.Arrange(new Rect(point, new Point(point.X + child.DesiredSize.Width, point.Y + child.DesiredSize.Height)));

                    if (child.DesiredSize.Width > largestWidth)
                        largestWidth = child.DesiredSize.Width;

                    point.Y = point.Y + child.DesiredSize.Height;

                    if ((i + 1) < Children.Count)
                    {
                        if ((point.Y + Children[i + 1].DesiredSize.Height) > finalSize.Height)
                        {
                            point.Y = 0;
                            point.X = point.X + largestWidth;
                            largestWidth = 0.0;
                        }
                    }

                    i++;
                }
            }

            return base.ArrangeOverride(finalSize);
        }
    }
}
