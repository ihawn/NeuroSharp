using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Data
{
    public static class DataUtils
    {
        public static Vector<double> OneHotEncodeVector(Vector<double> x, int categoryCount)
        {
            Vector<double> output = Vector<double>.Build.Dense(x.Count * categoryCount);

            for (int i = 0; i < x.Count; i++)
                output[i * categoryCount + (int)Math.Round(x[i])] = 1;

            return output;
        }
        
        public static Bitmap ScaleImage(Bitmap bmp, int maxWidth, int maxHeight)
        {
            var ratioX = (double)maxWidth / bmp.Width;
            var ratioY = (double)maxHeight / bmp.Height;
            var ratio = Math.Min(ratioX, ratioY);

            var newWidth = (int)(bmp.Width * ratio);
            var newHeight = (int)(bmp.Height * ratio);

            var newImage = new Bitmap(newWidth, newHeight);

            using(var graphics = Graphics.FromImage(newImage))
                graphics.DrawImage(bmp, 0, 0, newWidth, newHeight);

            return newImage;
        }
    }
}