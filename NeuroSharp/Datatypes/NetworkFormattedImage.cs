﻿using MathNet.Numerics.LinearAlgebra;
using System.Drawing;
using NeuroSharp.Utilities;

namespace NeuroSharp.Datatypes
{
    public struct NetworkFormattedImage
    {
        public Bitmap Image { get; set; }
        public Vector<double> Label { get; set; }

        public Matrix<double> Red { get; set; }
        public Matrix<double> Green { get; set; }
        public Matrix<double> Blue { get; set; }
        public Matrix<double> Gray { get; set; }
        public bool IsColor { get; set; }

        public NetworkFormattedImage(Bitmap image, Vector<double> label = null, bool isColor = true)
        {
            Image = image;
            Label = label;

            IsColor = isColor;
            
            Red = Matrix<double>.Build.Dense(image.Width, image.Height);
            Green = Matrix<double>.Build.Dense(image.Width, image.Height);
            Blue = Matrix<double>.Build.Dense(image.Width, image.Height);
            Gray = Matrix<double>.Build.Dense(image.Width, image.Height);

            for(int y = 0; y < image.Height; y++)
            {
                for(int x = 0; x < image.Width; x++)
                {
                    Color color = image.GetPixel(x, y);

                    if (isColor)
                    {
                        Red[x, y] = color.R/255d;
                        Green[x, y] = color.G/255d;
                        Blue[x, y] = color.B/255d;
                    }
                    else
                        Gray[x, y] = (color.R + color.G + color.B) / (255d * 3);
                }    
            }
        }

        public Vector<double> GetFlattenedImage()
        {
            return IsColor ? Vector<double>.Build.DenseOfEnumerable(MathUtils.Flatten(Red).ToList().Concat(
                                                                        MathUtils.Flatten(Green).ToList().Concat(
                                                                            MathUtils.Flatten(Blue)))) 
                                                                    : MathUtils.Flatten(Gray);
        }
    }
}
